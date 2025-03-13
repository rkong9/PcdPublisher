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


typedef pcl::PointCloud<pcl::PointXYZI> PCloudXYZI;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PCloudXYZIPtr;

class FeatureMapping {
public:
    FeatureMapping();
    ~FeatureMapping();
    void setCornerFeatures(PCloudXYZIPtr &pCorner) {
        mCornersCurr = pCorner;
    }
    void setSurfaceFeatures(PCloudXYZIPtr &pSurface) {
        mSurfaceCurr = pSurface;
    }

    int calcTransform();
    void readConfig(const YAML::Node &config);
    void updateFeatures();
    void getCalculatedFeat(PCloudXYZIPtr &pCorners, PCloudXYZIPtr &pSurfaces) {
      *pCorners = *mCalculatedCorner;
      *pSurfaces = *mCalculatedSurface;
    }

private:
    std::deque<PCloudXYZIPtr> mvHistoryCorner;
    std::deque<PCloudXYZIPtr> mvHistorySurface;

    PCloudXYZIPtr mCornersAll;
    PCloudXYZIPtr mSurfaceAll;

    PCloudXYZIPtr mCornersCurr;
    PCloudXYZIPtr mSurfaceCurr;

    PCloudXYZIPtr mCalculatedCorner;
    PCloudXYZIPtr mCalculatedSurface;

    pcl::KdTreeFLANN<pcl::PointXYZI> mCornersKdTree;
    pcl::KdTreeFLANN<pcl::PointXYZI> mSurfacesKdTree;

    std::mutex mCornerMtx;
    std::mutex mSurfaceMtx;
};

#endif

