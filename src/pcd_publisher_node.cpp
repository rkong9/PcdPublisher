#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <filesystem>
#include <algorithm>
#include <vector>

uint32_t mSeqId(0);
void publish_pcd(const ros::Publisher& pub, const std::string& pcd_file_path)
{
    // 读取 PCD 文件
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path, *cloud) == -1)
    {
        ROS_ERROR("Couldn't read file %s", pcd_file_path.c_str());
        return;
    }

    // 创建 PointCloud2 消息
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);

    // 设置消息头部信息
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "map";  // 设置坐标系为 "map"
    output.header.seq = mSeqId;

    // 发布消息
    pub.publish(output);
    ROS_INFO("Published PCD file: %s, seq:%d", pcd_file_path.c_str(), mSeqId);
    mSeqId++;
}

namespace fs = std::filesystem;
int main(int argc, char** argv)
{
    // 初始化 ROS 节点
    ros::init(argc, argv, "pcd_publisher_node");
    ros::NodeHandle nh;

    // 读取 PCD 文件的路径
    std::string dirPath;
    ros::NodeHandle nh_private("~");
    nh_private.getParam("pcd_dir", dirPath);
    double frame_rate(-1);
    nh_private.getParam("frame_rate", frame_rate);
    ROS_INFO("Get pcd dir path: %s", dirPath.c_str());
    if (!std::filesystem::exists(dirPath)) {
        ROS_ERROR("Error: Folder %s does not exist!", dirPath.c_str());
        return 1;
    }

    std::vector<std::string> vFilePath;

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pcd") {
            vFilePath.push_back(entry.path().stem().string());
        }
    }
    std::sort(vFilePath.begin(), vFilePath.end());

    // 创建 PointCloud2 消息发布者
    ros::Publisher pcd_pub =
        nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 10);

    size_t index(0);
    int64_t lastMs(0);
    ros::Rate rate(10);
    if (frame_rate > 0) {
        rate = ros::Rate(frame_rate);
    }
    while (ros::ok())
    {
        if (index < vFilePath.size()) {
            std::string currFileName = vFilePath[index];
            int64_t currTimeMs = std::atol(currFileName.c_str());
            if (lastMs == 0) {
                lastMs = currTimeMs;
            }

            int64_t sleepTime = currTimeMs - lastMs;
            if (frame_rate < 0) {
                if (sleepTime >= 30) {
                    ROS_INFO("pcd file index: %lu, sleep time:%ld curr time:%ld, last time:%ld", index, sleepTime, currTimeMs, lastMs);
                    ros::Duration(static_cast<float>(sleepTime) / 1000.0).sleep();
                }
            } else {
                rate.sleep();
            }

            // ros::Duration(0.5).sleep();
            std::string filePath = dirPath + '/' + currFileName + ".pcd";
            publish_pcd(pcd_pub, filePath);

            lastMs = currTimeMs;
            index++;
        }
        ros::spinOnce();
    }

    return 0;
}

