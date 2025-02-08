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

void processThread() {
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
        // pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudTrans(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudOut(new pcl::PointCloud<pcl::PointXYZI>());

        pcl::fromROSMsg(*pMsg, *pCloudIn);
    }
}

int main(int argc, char **argv) {
        // 初始化 ROS
    ros::init(argc, argv, "livox_feature_extractor");
    ros::NodeHandle nh;

    std::string cloudSource;

    // ros::NodeHandle nh_private("~");
    // nh_private.getParam("pcd_source", cloudSource);
    // mGridRes = nh_private.param("grid_res", 0.5);
    // mHeightTh = nh_private.param("height_th", 0.5);
    // mGetMinZFrames = nh_private.param("get_min_z_frames", 10);
    // ROS_INFO("Get pcd cloud source msg: %s", cloudSource.c_str());
    ros::Subscriber pc_sub = nh.subscribe(cloudSource, 10, pointCloudCallback);

    std::thread process(processThread);

    ros::spin();
    process.join();

    return 0;
}
