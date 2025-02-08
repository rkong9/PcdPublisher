#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <vector>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <thread>
#include <map>
#include <chrono>
#include <fstream>

std::mutex mDataMutex;
std::condition_variable mDataCv;
std::deque<sensor_msgs::PointCloud2Ptr> mqData;
std::deque<std::vector<float>> mqMinZ;
Eigen::Matrix4f mC2W(Eigen::Matrix4f::Identity()); // curr to word

float mGridRes;
float mHeightTh;
int mGetMinZFrames;

void cloudProcess(pcl::PointCloud<pcl::PointXYZI>::Ptr &pcloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &pout,
    std::vector<float> &vFinalZ, float res, float heightTh) {
    // 处理点云数据（例如使用 PassThrough 滤波器）
    pcl::PointCloud<pcl::PointXYZI>::Ptr pTemp(
        new pcl::PointCloud<pcl::PointXYZI>());

    Eigen::Vector4f minRoi(0.2, -10.0, -2.0, 1);
    Eigen::Vector4f maxRoi(60.0, 10.0, 2.0, 1);
    pcl::CropBox<pcl::PointXYZI> crop;
    crop.setMin(minRoi);
    crop.setMax(maxRoi);
    crop.setInputCloud(pcloud);
    crop.filter(*pTemp);

    for (auto pt : pTemp->points) {
        int index = static_cast<int>(std::floor(pt.x / res));
        if (index >= 0 && index < vFinalZ.size() &&
            vFinalZ[index] + heightTh < pt.z) {
            pout->push_back(pt);
        }
    }
    ROS_INFO("process pcd cloud src pts:%lu, tempPts%lu, outPts:%lu",
        pcloud->size(), pTemp->size(), pout->size());
}

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    sensor_msgs::PointCloud2Ptr pMsg(new sensor_msgs::PointCloud2);
    *pMsg = *msg;
    ROS_INFO("Get pcd cloud seq: %d", pMsg->header.seq);
    // 将 PointCloud2 消息转换为 PCL 点云
    std::unique_lock<std::mutex> lock(mDataMutex);
    mqData.push_back(pMsg);
    if (mqData.size() > 30) {
        mqData.pop_front();
    }
    mDataCv.notify_all();
}

std::vector<float> calcMinZOneFrame(float end, float res,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &pCloudIn) {
    int len = static_cast<int>(std::floor(end / res));
    std::vector<float> vMinZ(len, 0);
    for (auto &pt : pCloudIn->points) {
        int index = static_cast<int>(std::floor(pt.x / res));
        if (index < 0 || index >= static_cast<int>(vMinZ.size())) {
            continue;
        }
        if (pt.z < vMinZ[index]) {
            vMinZ[index] = pt.z;
        }
    }
    return vMinZ;
}

std::vector<float> calcMinZFromHistory(std::deque<std::vector<float>> &qMinZ) {
    std::vector<float> vFinelZ = qMinZ.front();
    for (auto &vMin : qMinZ) {
        for (size_t i = 0; i < vFinelZ.size(); i++) {
            if (vFinelZ[i] > vMin[i]) {
                vFinelZ[i] = vMin[i];
            }
        }
    }
    return vFinelZ;
}

void pointProcess2D(pcl::PointCloud<pcl::PointXYZ>::Ptr &pIn,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &pOut) {
    std::map<int, pcl::PointXYZ> mPt;
    for (auto &pt : pIn->points) {
        if (std::fabs(pt.x) < 0.05) {
            continue;
        }
        float deg10 = pcl::rad2deg(std::atan(pt.y / pt.x)) * 100;
        int degI = static_cast<int>(std::round(deg10));
        if (mPt.find(degI) == mPt.end()) {
            mPt[degI] = pt;
        } else {
            auto &pt2 = mPt[deg10];
            Eigen::Vector2f p(pt.x, pt.y);
            Eigen::Vector2f p2(pt2.x, pt2.y);
            if (p.norm() < p2.norm()) {
                mPt[deg10] = pt;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pTemp(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<pcl::PointXYZ> vPt;
    static int ci(0);
    std::ofstream of("./ang" + std::to_string(ci) + ".txt");
    for (auto &pair : mPt) {
        vPt.push_back(pair.second);
        if (of.is_open()) {
            of << std::fixed << std::setprecision(4) << pair.first << ",";
        }
    }
    ci++;
    of.close();

    // filter 2d cloud
    for (int i = 2; i < static_cast<int>(vPt.size()) - 2; i++) {

        auto &beg = vPt[i - 1];
        auto &end = vPt[i + 1];
        Eigen::Vector2f b1(beg.x, beg.y);
        Eigen::Vector2f curr(vPt[i].x, vPt[i].y);
        Eigen::Vector2f e1(end.x, end.y);
        float th(0.15);
        if ((curr - b1).norm() < th && (curr - e1).norm() < th) {
            pOut->push_back(vPt[i]);
        }
        // for (int j = i - 2; j <= i + 2; j++) {
        //     pOut->push_back(vPt[i]);
        // }
    }
}

bool lidarOdometry2D(pcl::PointCloud<pcl::PointXYZ>::Ptr &pLast,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &pCurr,
    Eigen::Matrix4f &transform) {

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(pLast);
    icp.setInputTarget(pCurr);
    pcl::PointCloud<pcl::PointXYZ> final;
    icp.align(final);
    if (!icp.hasConverged()) {
        ROS_WARN("icp did not converge!");
        return false;
    }
    transform = icp.getFinalTransformation();
    return true;
    // Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
    // Eigen::Vector3f trans = transform.block<3, 1>(0, 3);
}

void processThread() {
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    float leaf = nh_private.param("leaf_2d", 0.05);

    ros::Publisher processed_pc_pub =
        nh.advertise<sensor_msgs::PointCloud2>("/processed_point_cloud", 10);
    ros::Publisher processed_pc_pub_2d =
        nh.advertise<sensor_msgs::PointCloud2>("/processed_point_cloud_2d", 10);
    ros::Publisher pub_map =
        nh.advertise<sensor_msgs::PointCloud2>("/map", 2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pMap;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pLastFrame(new pcl::PointCloud<pcl::PointXYZ>());

    while (ros::ok())
    {
        sensor_msgs::PointCloud2Ptr pMsg(nullptr);
        {
            std::unique_lock<std::mutex> lock(mDataMutex);
            if (mqData.size() == 0) {
                if (mDataCv.wait_for(lock, std::chrono::milliseconds(500)) ==
                    std::cv_status::timeout) {
                    ROS_WARN("wait pcd cloud overtime...");
                    continue;
                }
            }
            pMsg = mqData.front();
            mqData.pop_front();
        }

        if (!pMsg) {
            ROS_WARN("invalid point cloud");
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudIn(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudTrans(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudOut(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::fromROSMsg(*pMsg, *pCloudIn);
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform(1, 1) = -1;
        transform(2, 2) = -1;
        pcl::transformPointCloud(*pCloudIn, *pCloudTrans, transform);

        std::vector<float> vMinZ = calcMinZOneFrame(60.0, mGridRes, pCloudTrans);
        mqMinZ.push_back(vMinZ);
        if (mqMinZ.size() < mGetMinZFrames) {
            continue;
        }

        std::vector<float> vFinalZ = calcMinZFromHistory(mqMinZ);
        std::stringstream ss;
        for (auto &Z : vFinalZ) {
            ss << std::fixed << std::setprecision(3) << Z << ", ";
        }
        ROS_INFO("minZ: %s", ss.str().c_str());
        cloudProcess(pCloudTrans, pCloudOut, vFinalZ, mGridRes, mHeightTh);

        mqMinZ.pop_front();
        ROS_INFO("process pcd cloud seq: %d, mqMinZ:%lu", pMsg->header.seq, mqMinZ.size());

        sensor_msgs::PointCloud2 processed_msg;
        pcl::toROSMsg(*pCloudOut, processed_msg);
        processed_msg.header = pMsg->header;
        processed_pc_pub.publish(processed_msg);

        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudOut2D(new pcl::PointCloud<pcl::PointXYZI>());
        for (auto pt: pCloudOut->points) {
            pt.z = 0.0;
            pCloudOut2D->push_back(pt);
        }
        pcl::VoxelGrid<pcl::PointXYZI> vog;
        vog.setInputCloud(pCloudOut2D);
        vog.setLeafSize(leaf, leaf, leaf);
        vog.filter(*pCloudOut2D);

        sensor_msgs::PointCloud2 processed_msg_2d;
        pcl::toROSMsg(*pCloudOut2D, processed_msg_2d);
        processed_msg_2d.header = pMsg->header;
        processed_pc_pub_2d.publish(processed_msg_2d);

        // if (!pMap) {
        //     pcl::PointCloud<pcl::PointXYZ>::Ptr pTemp(new pcl::PointCloud<pcl::PointXYZ>());
        //     pMap = pTemp;
        //     pcl::copyPointCloud(*pCloudOut2D, *pMap);
        //     continue;
        // }

        transform = Eigen::Matrix4f::Identity();
        pcl::PointCloud<pcl::PointXYZ>::Ptr pSrc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr pTransTemp(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr pF2Map(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*pCloudOut2D, *pSrc);
        pointProcess2D(pSrc, pTransTemp);
        pSrc = pTransTemp;
        // pcl::transformPointCloud(*pSrc, *pTransTemp, mC2W);
        if (!pLastFrame) {
            pLastFrame = pSrc;
            continue;
        }
        bool ret = lidarOdometry2D(pLastFrame, pSrc, transform);
        sensor_msgs::PointCloud2 map2d;
        pcl::toROSMsg(*pSrc, map2d);
        map2d.header = pMsg->header;
        pub_map.publish(map2d);

        if (ret) {
            Eigen::Vector3f trans = transform.block<3, 1>(0, 3);
            ROS_INFO("get trans:%.5f, %.5f, %.5f", trans.x(), trans.y(), trans.z());
        } else {
            ROS_WARN("get trans of two frame failed, seq:%d", pMsg->header.seq);
        }

        pLastFrame = pSrc;

        // bool ret = lidarOdometry2D(pTransTemp, pMap, transform);
        // if (ret) {
        //     mC2W = transform * mC2W;
        //     pcl::transformPointCloud(*pSrc, *pF2Map, mC2W);
        //     *pMap += *pF2Map;
        //     pcl::VoxelGrid<pcl::PointXYZ> vogM;
        //     vogM.setInputCloud(pMap);
        //     vogM.setLeafSize(leaf, leaf, leaf);
        //     vogM.filter(*pMap);

        //     sensor_msgs::PointCloud2 map2d;
        //     pcl::toROSMsg(*pMap, map2d);
        //     map2d.header = pMsg->header;
        //     pub_map.publish(map2d);
        //     // MM_WARN("get trans of two frame failed");
        // } else {
        //     ROS_WARN("get trans of two frame failed, seq:%d", pMsg->header.seq);
        // }
    }
}

int main(int argc, char **argv) {
        // 初始化 ROS
    ros::init(argc, argv, "pcd_publisher_process");
    ros::NodeHandle nh;

    ros::NodeHandle nh_private("~");
    std::string cloudSource;
    nh_private.getParam("pcd_source", cloudSource);
    mGridRes = nh_private.param("grid_res", 0.5);
    mHeightTh = nh_private.param("height_th", 0.5);
    mGetMinZFrames = nh_private.param("get_min_z_frames", 10);
    ROS_INFO("Get pcd cloud source msg: %s", cloudSource.c_str());
    ros::Subscriber pc_sub = nh.subscribe(cloudSource, 10, pointCloudCallback);

    std::thread process(processThread);

    ros::spin();
    process.join();

    return 0;
}
