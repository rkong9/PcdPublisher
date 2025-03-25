#ifndef _LIVOX_FEATURE_EXTRACTOR_HPP_
#define _LIVOX_FEATURE_EXTRACTOR_HPP_
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

typedef pcl::PointCloud<pcl::PointXYZI> PCloudXYZI;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PCloudXYZIPtr;

class LivoxFeature {
public:
  enum E_point_type {
    e_pt_normal = 0,                      // normal points
    e_pt_000 = 0x0001 << 0,               // points [0,0,0]
    e_pt_too_near = 0x0001 << 1,          // points in short range
    e_pt_reflectivity_low = 0x0001 << 2,  // low reflectivity
    e_pt_reflectivity_high = 0x0001 << 3, // high reflectivity
    e_pt_circle_edge = 0x0001 << 4,       // points near the edge of circle
    e_pt_nan = 0x0001 << 5,               // points with infinite value
    e_pt_small_view_angle = 0x0001 << 6,  // points with large viewed angle
  };

  enum E_feature_type // if and only if normal point can be labeled
  { e_label_invalid = 0,
    e_label_unlabeled = 1,
    e_label_corner = 0x0001 << 2,
    e_label_surface = 0x0001 << 3,
    e_label_near_nan = 0x0001 << 4,
    e_label_near_zero = 0x0001 << 5,
    e_label_hight_intensity = 0x0001 << 6,
  };

  // Encode point infos using points intensity, which is more convenient for
  // debugging.
  enum E_intensity_type {
    e_I_raw = 0,
    e_I_motion_blur,
    e_I_motion_mix,
    e_I_sigma,
    e_I_scan_angle,
    e_I_curvature,
    e_I_view_angle,
    e_I_time_stamp
  };

  struct Pt_infos {
    int pt_type = e_pt_normal;
    int pt_label = e_label_unlabeled;
    int idx = 0;
    int scan_idx = 0;
    float raw_intensity = 0.f;
    float time_stamp = 0.0;
    float polar_angle = 0.f;
    int polar_direction = 0;
    float polar_dis_sq2 = 0.f;
    float depth_sq2 = 0.f;
    float curvature = 0.0;
    float view_angle = 0.0;
    float sigma = 0.0;
    Eigen::Matrix<float, 2, 1> pt_2d_img; // project to X==1 plane
  };

  LivoxFeature();
  ~LivoxFeature();
  void getFeatures(pcl::PointCloud<pcl::PointXYZI> &pc_corners,
                   pcl::PointCloud<pcl::PointXYZI> &pc_surface,
                   pcl::PointCloud<pcl::PointXYZI> &pc_full_res, size_t begIdx,
                   size_t endIdx, int color_mode);

  std::vector<PCloudXYZIPtr> processCloud(PCloudXYZIPtr &pCloudIn);
  void set_intensity(pcl::PointXYZI &pt, int p_idx,
                     const E_intensity_type &i_type = e_I_motion_blur);
  void readConfig(const YAML::Node &config);

private:
  // void extract_good_points();
  void add_mask_of_point(Pt_infos *pt_infos, const E_point_type &pt_type,
                         int neighbor_count = 0);
  std::vector<PCloudXYZIPtr> preprocess_pointinfo(PCloudXYZI &cloudIn);

  void compute_features();

private:
  pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudIn;
  std::vector<Pt_infos> m_pts_info_vec;
  std::vector<pcl::PointXYZI> m_raw_pts_vec;
  size_t m_input_points_size;
  float m_livox_min_allow_dis = 1.0;
  float m_livox_min_sigma = 7e-3;
  float m_max_edge_polar_pos;
  float m_feature_surface_depth_sq2_th = 40000.0;
  float m_feature_corner_depth_sq2_th = 6400.0;
  float m_minimum_view_angle;
  float m_max_surface_curvature_th;
  float m_min_corner_curvature_th;
  float m_corner_neighbour_diff_th;
  int frame_cnt;
  int64_t m_timestamp;

  // config
};

inline Eigen::Vector3f getEPt(const pcl::PointXYZI &pt) {
    return Eigen::Vector3f(pt.x, pt.y, pt.z);
}

// 1. 角特征提取
// 2. 平面特征提取
// 3. ceres优化
// 4. o发布

// --- 
// 滑动窗口
// 性能优化


// new Feature extractor
// -> corner features
//     1. neighboure 全部是有效点
//     2. 归一化距离计算
//     3. selected 角点距离需要离激光更近
//     4. 一个scan中仅选取最大曲率的点
//     5. sequence check
//     6. x-axis gradient
// 
//     // todo
//     1. - 强度diff计算
//     2. X-axis 增强，设置权重
//     3. view angle check
// 
// -> surfaces features
//     1.neighboure 全部是有效点
//     2. 相邻点距离小于某个值
//     3. 曲率小于某个值
//     4. 距离梯度小于某个值

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();
    void setInputCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pInCloud) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr temp(new pcl::PointCloud<pcl::PointXYZI>());
        *temp = *pInCloud;
        mpSrcCloud = temp;
    }
    void compute_features();
    int getFeatures(pcl::PointCloud<pcl::PointXYZI>::Ptr &pCorner,
        pcl::PointCloud<pcl::PointXYZI>::Ptr &pSurface, pcl::PointCloud<pcl::PointXYZI>::Ptr &pValidAll);
    void readConfig(const YAML::Node &config);

    enum class PtLabels{
        pt_nan = 1,
        pt_too_near = 1 << 1,
        pt_corner = 1 << 2,
        pt_surface = 1 << 3,
    };


    struct PtsInfo {
        Eigen::Vector3f v2Next;
        float dis2Next;
        Eigen::Vector3f vCurv3f;
        float curv2;
        float dis2Origin;
        float minNeighbourDiff;
        float maxNeighbourDiff;
        int validNears;
        int disSmallNears;
        int disStableNums;
        int scanid;
        int idx;
        int label;

        PtsInfo () {
            v2Next = Eigen::Vector3f::Zero();
            vCurv3f = Eigen::Vector3f::Zero();
            dis2Next = 0.0;
            curv2 = 0.0;
            dis2Origin = 0.0;
            minNeighbourDiff = 1000.0;
            maxNeighbourDiff = 0.0;
            validNears = 0;
            disSmallNears = 0;
            disStableNums = 0;
            scanid = -1;
            idx = -1;
            label = -1;
        }

        friend std::ostream& operator<<(std::ostream& os, const PtsInfo& pti) {
            os << "vn:" << pti.v2Next.transpose() << "," << pti.dis2Next << ", cur:" << pti.vCurv3f.transpose() <<
                ",   | " << pti.curv2 << " |, dis:" << pti.dis2Origin << ", min-max" << pti.minNeighbourDiff << "," <<
                pti.maxNeighbourDiff << ", " << pti.validNears << "," << pti.disSmallNears << "," << pti.disStableNums
                << "," << pti.label;
            return os;
        }
    };

private:
    inline int getPointStatus(const pcl::PointXYZI &pt);

private:
    int mDebugLevel;
    pcl::PointCloud<pcl::PointXYZI>::Ptr mpSrcCloud;
    std::vector<PtsInfo> mvPtsInfo;
    int mPtsNums;

    // config
    int mPtsPerSec;
    int mCalcWinSize;
    float mSmallDisTh;
    float mZeroPtTh;
    float mCornerCurvTh;
    float mMaxCornerDis;
    int mCornerPtsInterval;

    float mSurfCurvTh;
    float mMaxSurfDis;
    int mSurfPtsInterval;
    float mSurfNeighbourDiffRatio;
};


#endif
