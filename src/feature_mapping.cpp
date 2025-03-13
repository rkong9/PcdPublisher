#include "feature_mapping.hpp"

FeatureMapping::FeatureMapping() {

}

FeatureMapping::~FeatureMapping() {

}

int FeatureMapping::calcTransform(Eigen::Matrix4d &trans) {
    float radius(0.4);
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    int validLines(0);
    for (auto &pt : mCornersCurr->points) {
        int n(0);
        {
            std::unique_lock lock(mCornerMtx);
            n = mCornersKdTree.radiusSearch(pt, radius, pointIdxNKNSearch, pointNKNSquaredDistance);
        }

        if (n < 2) {
            continue;
        }

        // get line
        // add line ceres block
    }

    int validSurfaces;
    for (auto &pt : mSurfaceCurr->points) {
        int n(0);
        {
            std::unique_lock lock(mCornerMtx);
            n = mSurfacesKdTree.radiusSearch(pt, radius, pointIdxNKNSearch, pointNKNSquaredDistance);
        }

        if (n < 4) {
            continue;
        }

        // get plane paras
        // add plane ceres block
    }

    return 0;
}

void FeatureMapping::updateFeatures() {
    PCloudXYZIPtr pFeatCorners(new PCloudXYZI);
    PCloudXYZIPtr pFeatSurfaces(new PCloudXYZI);
    for (auto &it = mvHistoryCorner.rbegin(); it != mvHistoryCorner.rend(); it++) {
        *pFeatCorners += *(*it);
    }

    {
      std::unique_lock lock(mCornerMtx);
      mCornersAll = pFeatCorners;
      mCornersKdTree.setInputCloud(mCornersAll);
    }

    for (auto &it = mvHistorySurface.rbegin(); it != mvHistorySurface.rend(); it++) {
        *pFeatSurfaces += *(*it);
    }

    {
        std::unique_lock lock(mSurfaceMtx);
        mSurfaceCurr = pFeatSurfaces;
        mSurfacesKdTree.setInputCloud(mSurfaceCurr);
    }
}

