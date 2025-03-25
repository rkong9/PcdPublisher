#ifndef _FEATURE_MAPPING_HPP_
#define _FEATURE_MAPPING_HPP_

#include <map>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <pcl/search/kdtree.h>
#include <ceres/ceres.h>


typedef pcl::PointCloud<pcl::PointXYZI> PCloudXYZI;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PCloudXYZIPtr;

class FeatureMapping {
public:
    FeatureMapping();
    ~FeatureMapping();
    void setInputData(PCloudXYZIPtr &pCorners, PCloudXYZIPtr &pSurfaces, PCloudXYZIPtr &pValidAll);

    int calcTransform(Eigen::Quaterniond &q, Eigen::Vector3d &trans);
    void readConfig(const YAML::Node &config);
    void updateAllFeatures();
    void reset();
    void trans2End();
    void getCalculatedFeature(PCloudXYZIPtr &pCorners, PCloudXYZIPtr &pSurfaces) {
      *pCorners = *mCalculatedCorner;
      *pSurfaces = *mCalculatedSurface;
    }
    void getOdom(Eigen::Quaterniond &q, Eigen::Vector4d &t);
    void assocateCloud2Map(PCloudXYZIPtr &pcloudIn);
    void getMap();

private:
    int buildPoint2LineProblem(ceres::Problem &problem);
    int buildPoint2PlaneProblem(ceres::Problem &problem);
    Eigen::Matrix3d getCovMat(const PCloudXYZIPtr &pcloud, const std::vector<int> &indices);
    inline Eigen::Vector3d getEPointFromP(const pcl::PointXYZI &pt);

private:
    // data buffer
    std::deque<PCloudXYZIPtr> mvHistoryCorner;
    std::deque<PCloudXYZIPtr> mvHistorySurface;

    PCloudXYZIPtr mCornersAll;
    PCloudXYZIPtr mSurfaceAll;

    PCloudXYZIPtr mCornersCurr;
    PCloudXYZIPtr mSurfaceCurr;
    PCloudXYZIPtr mValidAllCurr;
    PCloudXYZIPtr mMap;

    PCloudXYZIPtr mCalculatedCorner;
    PCloudXYZIPtr mCalculatedSurface;

    pcl::KdTreeFLANN<pcl::PointXYZI> mCornersKdTree;
    pcl::KdTreeFLANN<pcl::PointXYZI> mSurfacesKdTree;

    std::vector<int> mSearchLineIndices;
    std::vector<int> mSearchSurfIndices;

    PCloudXYZIPtr mSearchLineClouds;
    PCloudXYZIPtr mSearchSurClouds;

    std::mutex mCornerMtx;
    std::mutex mSurfaceMtx;

    // config
    size_t mBufferSize;

    // int mValidLines;
    // int mValidSurfaces;

    // calc transform
    int mSearchLinePts;
    int mValidLinePtsTh;
    float mSearchLineAngleTh;
    float mLineHuberLossTh;
    float mLineCheckRatio;

    int mSearchSurfacePts;
    int mValidSurfPtsTh;
    float mSearchSurAngleTh;
    float mSurfHuberLossTh;
    float mSurfCheckRatio[2];

    Eigen::Quaterniond m_q_w_last;
    Eigen::Vector3d m_t_w_last;

    double m_para_buffer[7] = {0, 0, 0, 1, 0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> m_q_incre = Eigen::Map<Eigen::Quaterniond>(m_para_buffer);
    Eigen::Map<Eigen::Vector3d> m_t_incre = Eigen::Map<Eigen::Vector3d>(m_para_buffer + 4);
};

struct DataPack {
    sensor_msgs::PointCloud2Ptr pCorners;
    sensor_msgs::PointCloud2Ptr pSurfaces;
    sensor_msgs::PointCloud2Ptr pValidAll;
    DataPack() {
        pCorners = nullptr;
        pSurfaces = nullptr;
        pValidAll = nullptr;
    }

    bool isComplete() {
        return pCorners && pSurfaces && pValidAll;
    }
};

#endif

