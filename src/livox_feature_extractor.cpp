#include <Eigen/Dense>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <vector>
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "livox_feature_extractor.hpp"


std::mutex mDataMutex;
std::condition_variable mDataCv;
std::deque<sensor_msgs::PointCloud2Ptr> mqData;

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
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
  LivoxFeature lf;

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  std::string configPath;
  nh_private.getParam("config_path", configPath);
  lf.readConfig(configPath);

  ros::Publisher corner_pc_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/corner_pc", 10);
  ros::Publisher surface_pc_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/surface_pc", 10);
  ros::Publisher all_pc_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/all_pc_pub", 10);

  while (ros::ok()) {
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

    pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudIn(
        new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudTrans(new
    // pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudOut(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudCorner(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudSurface(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudFullAll(
        new pcl::PointCloud<pcl::PointXYZI>());

    pcl::fromROSMsg(*pMsg, *pCloudIn);

    std::vector<PCloudXYZIPtr> vScansClouds = lf.processCloud(pCloudIn);
    lf.getFeatures(*pCloudCorner, *pCloudSurface, *pCloudFullAll, 0, pCloudIn->size() - 1, 5);

    sensor_msgs::PointCloud2 corner_msg;
    pcl::toROSMsg(*pCloudCorner, corner_msg);
    corner_msg.header = pMsg->header;
    corner_pc_pub.publish(corner_msg);

    sensor_msgs::PointCloud2 surface_msg;
    pcl::toROSMsg(*pCloudSurface, surface_msg);
    surface_msg.header = pMsg->header;
    surface_pc_pub.publish(surface_msg);

    sensor_msgs::PointCloud2 FullAll_msg;
    pcl::toROSMsg(*pCloudFullAll, FullAll_msg);
    FullAll_msg.header = pMsg->header;
    all_pc_pub.publish(FullAll_msg);
    ROS_INFO("get %lu scans from frame, allpts:%lu, cornerPts:%lu, surfacePts:%lu, fullAllPts:%lu",
        vScansClouds.size(), pCloudIn->size(), pCloudCorner->size(),
        pCloudSurface->size(), pCloudFullAll->size());
  }
}

// void extract_good_points(pcl::PointCloud<pcl::PointXYZI>::Ptr pCloudIn,
//                          pcl::PointCloud<pcl::PointXYZI>::Ptr &pCloudOut) {
//   float corner_cav_th(0.05);
//   float surface_cav_th(0.05);
//   float max_fov(70.4); // degrees
// }

LivoxFeature::LivoxFeature() {
  // 57.3 是角度与弧度的转换单位 70.4 max_fov
  m_max_edge_polar_pos = std::pow(tan(33.3 / 57.3) * 1, 2);
  frame_cnt = 0;
}

LivoxFeature::~LivoxFeature() {}

void LivoxFeature::readConfig(const std::string &cfgPath) {
    YAML::Node config = YAML::LoadFile(cfgPath);
    float maxAngle = config["max_angle"].as<float>();
    m_max_edge_polar_pos = std::pow(tan(maxAngle / 57.3) * 1.0, 2);

    float maxSurfaceDepth = config["feature_max_surface_depth"].as<float>();
    m_feature_surface_depth_sq2_th = maxSurfaceDepth * maxSurfaceDepth;

    float maxCornerDepth = config["feature_max_corner_depth"].as<float>();
    m_feature_corner_depth_sq2_th = maxCornerDepth * maxCornerDepth;

    m_minimum_view_angle = config["min_view_angle"].as<float>();
    m_max_surface_curvature_th = config["max_surface_curvature_th"].as<float>();
    m_min_corner_curvature_th = config["min_corner_curvature_th"].as<float>();
    m_corner_neighbour_diff_th = config["corner_neighbour_diff_th"].as<float>();
}

void LivoxFeature::add_mask_of_point (Pt_infos *pt_infos,
                                     const E_point_type &pt_type,
                                     int neighbor_count) {

  int idx = pt_infos->idx;
  pt_infos->pt_type |= pt_type;

  if (neighbor_count > 0) {
    for (int i = -neighbor_count; i < neighbor_count; i++) {
      idx = pt_infos->idx + i;

      if (i != 0 && (idx >= 0) && (idx < (int)m_pts_info_vec.size())) {
        m_pts_info_vec[idx].pt_type |= pt_type;
      }
    }
  }
}

float dis2_xy(float x, float y) { return x * x + y * y; }

float depth2_xyz(float x, float y, float z) { return x * x + y * y + z * z; }

void LivoxFeature::set_intensity(
    pcl::PointXYZI &pt, int p_idx,
    const E_intensity_type &i_type) {
  Pt_infos *pt_info = &m_pts_info_vec[p_idx];
  switch (i_type) {
  case (e_I_raw):
    pt.intensity = pt_info->raw_intensity;
    break;
  case (e_I_motion_blur):
    pt.intensity = ((float)pt_info->idx) / (float)m_input_points_size;
    assert(pt.intensity <= 1.0 && pt.intensity >= 0.0);
    break;
  case (e_I_motion_mix):
    pt.intensity =
        0.1 * ((float)pt_info->idx + 1) / (float)m_input_points_size +
        (int)(pt_info->raw_intensity);
    break;
  case (e_I_scan_angle):
    pt.intensity = pt_info->polar_angle;
    break;
  case (e_I_curvature):
    pt.intensity = pt_info->curvature;
    break;
  case (e_I_view_angle):
    pt.intensity = pt_info->view_angle;
    break;
  case (e_I_time_stamp):
    pt.intensity = pt_info->time_stamp;
  default:
    pt.intensity = ((float)pt_info->idx + 1) / (float)m_input_points_size;
  }
  return;
}

std::vector<PCloudXYZIPtr>
LivoxFeature::preprocess_pointinfo(PCloudXYZI &cloudIn) {
  // step0. 从第二个点开始遍历每个点直至倒数第二个点
  // stpe1. 去除掉过近点和无效点后计算每个点的曲率(相邻点也不能是过近或者无效点)
  // step2.
  // 计算当前点到激光中心这个向量与当前点前后2个邻点组成的向量的夹角(入射角)
  // step3. 忽略掉入射角过小的点(阈值为10)
  // step4. 对于有效点如果曲率小于设定阈值则认为是平面
  // step5.
  // 如果曲率大于阈值并且前后点的距离差小于10%,则认为是角点(过滤掉遮挡的点)
  size_t pts_size = cloudIn.size();
  m_pts_info_vec.clear();
  m_pts_info_vec.resize(pts_size);
  m_raw_pts_vec.resize(pts_size);

  int last_split_idx(0);
  int scan_idx(0);

  std::vector<PCloudXYZIPtr> vScans;
  size_t curr_scan_pts(0);
  PCloudXYZIPtr curr_scan_clouds(new PCloudXYZI);
  curr_scan_clouds->points.resize(pts_size);

  for (size_t idx = 0; idx < pts_size; idx++) {
    m_raw_pts_vec[idx] = cloudIn.points[idx];
    Pt_infos *pt_info = &m_pts_info_vec[idx];
    pt_info->raw_intensity = cloudIn.points[idx].intensity;
    pt_info->idx = idx;
    // pt_info->time_stamp = m_current_time + ((float)idx) *
    // m_time_internal_pts;

    if (!std::isfinite(m_raw_pts_vec[idx].x) ||
        !std::isfinite(m_raw_pts_vec[idx].y) ||
        !std::isfinite(m_raw_pts_vec[idx].z)) {
      add_mask_of_point(pt_info, e_pt_nan);
      continue;
    }

    if (cloudIn.points[idx].x == 0) {
      if (idx == 0) {
        std::cout << "First point should be normal!!!" << std::endl;
        pt_info->pt_2d_img << 0.01, 0.01;
        pt_info->polar_dis_sq2 = 0.0001;
        add_mask_of_point(pt_info, e_pt_000);
      } else {
        pt_info->pt_2d_img = m_pts_info_vec[idx - 1].pt_2d_img;
        pt_info->polar_dis_sq2 = m_pts_info_vec[idx - 1].polar_dis_sq2;
        add_mask_of_point(pt_info, e_pt_000);
        continue;
      }
    }

    pt_info->depth_sq2 = depth2_xyz(
        cloudIn.points[idx].x, cloudIn.points[idx].y, cloudIn.points[idx].z);

    pt_info->pt_2d_img << cloudIn.points[idx].y / cloudIn.points[idx].x,
        cloudIn.points[idx].z / cloudIn.points[idx].x;
    pt_info->polar_dis_sq2 =
        dis2_xy(pt_info->pt_2d_img(0), pt_info->pt_2d_img(1));

    if (pt_info->depth_sq2 <
        m_livox_min_allow_dis * m_livox_min_allow_dis) // to close
    {
      add_mask_of_point(pt_info, e_pt_too_near);
    }

    pt_info->sigma = pt_info->raw_intensity / pt_info->polar_dis_sq2;

    if (pt_info->sigma < m_livox_min_sigma) {
      add_mask_of_point(pt_info, e_pt_reflectivity_low);
    }

    if (pt_info->polar_dis_sq2 > m_max_edge_polar_pos) {
      add_mask_of_point(pt_info, e_pt_circle_edge, 2);
    }

    // Split scans
    if (idx >= 1) {
      float dis_incre =
          pt_info->polar_dis_sq2 - m_pts_info_vec[idx - 1].polar_dis_sq2;

      if (dis_incre > 0) // far away from zero
      {
        pt_info->polar_direction = 1;
      }

      if (dis_incre < 0) // move toward zero
      {
        pt_info->polar_direction = -1;
      }

      if (pt_info->polar_direction * m_pts_info_vec[idx - 1].polar_direction <
          0) {
        if ((idx - last_split_idx) > 50 )
                {
            last_split_idx = idx;
            scan_idx++;
            curr_scan_clouds->points.resize(curr_scan_pts);
            vScans.push_back(curr_scan_clouds);

            PCloudXYZIPtr temp(new PCloudXYZI);
            temp->resize(pts_size);
            curr_scan_clouds = temp;
          }
      }

      pt_info->scan_idx = scan_idx;
      curr_scan_pts++;

      // // 经过边缘点并且距离上一个分割点之间的点数超过50个
      // if ( pt_info->polar_direction == -1 && m_pts_info_vec[ idx - 1
      // ].polar_direction == 1 )
      // {
      //     if (idx -  last_split_idx) > 50 )
      //     {
      //         split_idx.push_back( idx );
      //         edge_idx.push_back( idx );
      //         continue;
      //     }
      // }

      // // 经过原点并且距离上一个分割点之间的点数超过50个
      // if ( pt_info->polar_direction == 1 && m_pts_info_vec[ idx - 1
      // ].polar_direction == -1 )
      // {
      //     if ( zero_idx.size() == 0 || ( idx - split_idx[ split_idx.size() -
      //     1 ] ) > 50 )
      //     {
      //         split_idx.push_back( idx );

      //         zero_idx.push_back( idx );
      //         continue;
      //     }
      // }
    }
  }
  return vScans;
}

void LivoxFeature::compute_features() {
  unsigned int pts_size = m_raw_pts_vec.size();
  size_t curvature_ssd_size = 2;
  int critical_rm_point = e_pt_000 | e_pt_nan | e_pt_circle_edge;
  float neighbor_accumulate_xyz[3] = {0.0, 0.0, 0.0};

  std::string file_name = std::to_string(frame_cnt) + "_cloud.txt";
  std::ofstream of(file_name);

  for (size_t idx = curvature_ssd_size; idx < pts_size - curvature_ssd_size;
       idx++) {
    auto &pt = m_raw_pts_vec[idx];
    of << idx << "," << pt.x << "," << pt.y << "," << pt.z << ",";
    if (m_pts_info_vec[idx].pt_type & critical_rm_point) {
      of << "invalid \n";
      continue;
    }

    /*********** Compute curvate ************/
    neighbor_accumulate_xyz[0] = 0.0;
    neighbor_accumulate_xyz[1] = 0.0;
    neighbor_accumulate_xyz[2] = 0.0;

    for (size_t i = 1; i <= curvature_ssd_size; i++) {
      if ((m_pts_info_vec[idx + i].pt_type & e_pt_000) ||
          (m_pts_info_vec[idx - i].pt_type & e_pt_000)) {
        if (i == 1) {
          m_pts_info_vec[idx].pt_label |= e_label_near_zero;
        } else {
          m_pts_info_vec[idx].pt_label = e_label_invalid;
        }
        break;
      } else if ((m_pts_info_vec[idx + i].pt_type & e_pt_nan) ||
                 (m_pts_info_vec[idx - i].pt_type & e_pt_nan)) {
        if (i == 1) {
          m_pts_info_vec[idx].pt_label |= e_label_near_nan;
        } else {
          m_pts_info_vec[idx].pt_label = e_label_invalid;
        }
        break;
      } else {
        neighbor_accumulate_xyz[0] +=
            m_raw_pts_vec[idx + i].x + m_raw_pts_vec[idx - i].x;
        neighbor_accumulate_xyz[1] +=
            m_raw_pts_vec[idx + i].y + m_raw_pts_vec[idx - i].y;
        neighbor_accumulate_xyz[2] +=
            m_raw_pts_vec[idx + i].z + m_raw_pts_vec[idx - i].z;
      }
    }

    if (m_pts_info_vec[idx].pt_label == e_label_invalid) {
      of << "invalid\n";
      continue;
    }

    neighbor_accumulate_xyz[0] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].x;
    neighbor_accumulate_xyz[1] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].y;
    neighbor_accumulate_xyz[2] -= curvature_ssd_size * 2 * m_raw_pts_vec[idx].z;
    m_pts_info_vec[idx].curvature =
        neighbor_accumulate_xyz[0] * neighbor_accumulate_xyz[0] +
        neighbor_accumulate_xyz[1] * neighbor_accumulate_xyz[1] +
        neighbor_accumulate_xyz[2] * neighbor_accumulate_xyz[2];
    of << m_pts_info_vec[idx].curvature << ", ";

    /*********** Compute plane angle ************/
    Eigen::Matrix<float, 3, 1> vec_a(m_raw_pts_vec[idx].x, m_raw_pts_vec[idx].y,
                                     m_raw_pts_vec[idx].z);
    Eigen::Matrix<float, 3, 1> vec_b(
        m_raw_pts_vec[idx + curvature_ssd_size].x -
            m_raw_pts_vec[idx - curvature_ssd_size].x,
        m_raw_pts_vec[idx + curvature_ssd_size].y -
            m_raw_pts_vec[idx - curvature_ssd_size].y,
        m_raw_pts_vec[idx + curvature_ssd_size].z -
            m_raw_pts_vec[idx - curvature_ssd_size].z);
    float vec_a_norm = vec_a.norm();
    float vec_b_norm = vec_b.norm();
    if (std::fabs(vec_a_norm) < 1e-5 || std::fabs(vec_b_norm) < 1e-5) {
        m_pts_info_vec[idx].view_angle = 0.0;
    } else {
        m_pts_info_vec[idx].view_angle =
            acos( abs( vec_a.dot( vec_b ) ) / ( vec_a_norm * vec_b_norm) ) * 57.3;
            //Eigen_math::vector_angle(vec_a, vec_b, 1) * 57.3;
    }

    of << m_pts_info_vec[idx].depth_sq2 << ", ";
    of << m_pts_info_vec[idx].view_angle << ", ";
    of << m_pts_info_vec[idx].polar_dis_sq2 << ", ";
    of << m_pts_info_vec[idx].polar_direction << ", ";
    of << m_pts_info_vec[idx].scan_idx << ", ";


    // printf( "Idx = %d, angle = %.2f\r\n", idx,  m_pts_info_vec[ idx
    // ].view_angle );
    if (m_pts_info_vec[idx].view_angle > m_minimum_view_angle) {

      if (m_pts_info_vec[idx].curvature < m_max_surface_curvature_th) {
        m_pts_info_vec[idx].pt_label |= e_label_surface;
        of << "surface,";
      }

      float sq2_diff = m_corner_neighbour_diff_th;

      if (m_pts_info_vec[idx].curvature > m_min_corner_curvature_th) {
        if (m_pts_info_vec[idx].depth_sq2 <=
                m_pts_info_vec[idx - curvature_ssd_size].depth_sq2 &&
            m_pts_info_vec[idx].depth_sq2 <=
                m_pts_info_vec[idx + curvature_ssd_size].depth_sq2) {

          if (abs(m_pts_info_vec[idx].depth_sq2 -
                  m_pts_info_vec[idx - curvature_ssd_size].depth_sq2) <
                  sq2_diff * m_pts_info_vec[idx].depth_sq2 ||
              abs(m_pts_info_vec[idx].depth_sq2 -
                  m_pts_info_vec[idx + curvature_ssd_size].depth_sq2) <
                  sq2_diff * m_pts_info_vec[idx].depth_sq2) {
            m_pts_info_vec[idx].pt_label |= e_label_corner;
            // if (m_pts_info_vec[idx].depth_sq2 < 100.0) {
              of << " corner,";
            // }
          }
        }
      }
    }
    of << "\n";
  }
  of.close();
  frame_cnt++;
}

std::vector<PCloudXYZIPtr> LivoxFeature::processCloud(PCloudXYZIPtr &pCloudIn) {
  m_input_points_size = pCloudIn->points.size();

  std::vector<PCloudXYZIPtr> vClouds =
      preprocess_pointinfo(*pCloudIn);
  compute_features();

  return vClouds;
}

void LivoxFeature::getFeatures(pcl::PointCloud<pcl::PointXYZI> &pc_corners,
                               pcl::PointCloud<pcl::PointXYZI> &pc_surface,
                               pcl::PointCloud<pcl::PointXYZI> &pc_full_res,
                               size_t begIdx, size_t endIdx, int color_mode) {

  int corner_num = 0;
  int surface_num = 0;
  int full_num = 0;

  pc_corners.resize(m_pts_info_vec.size()); // 前后2次resize 效率会更高
  pc_surface.resize(m_pts_info_vec.size());
  pc_full_res.resize(m_pts_info_vec.size());

  int pt_critical_rm_mask = e_pt_000 | e_pt_nan | e_pt_too_near;

  for (size_t i = begIdx; i < endIdx; i++) {
    if ((m_pts_info_vec[i].pt_type & pt_critical_rm_mask) == 0) {
      if (m_pts_info_vec[i].pt_label & e_label_corner) {
        if (m_pts_info_vec[i].pt_type != e_pt_normal)
          continue;
        if (m_pts_info_vec[i].depth_sq2 < m_feature_corner_depth_sq2_th) {
          pc_corners.points[corner_num] = m_raw_pts_vec[i];
          pc_corners.points[corner_num].intensity =
              m_pts_info_vec[i].time_stamp;
          if (color_mode) {
            set_intensity(pc_corners.points[corner_num], i,
                          static_cast<E_intensity_type>(color_mode));
          }
          corner_num++;
        }
      }
      if (m_pts_info_vec[i].pt_label & e_label_surface) {
        if (m_pts_info_vec[i].depth_sq2 < m_feature_surface_depth_sq2_th) {
          pc_surface.points[surface_num] = m_raw_pts_vec[i];
          pc_surface.points[surface_num].intensity =
              float(m_pts_info_vec[i].time_stamp);
          if (color_mode) {
            set_intensity(pc_surface.points[surface_num], i,
                          static_cast<E_intensity_type>(color_mode));
          }
          surface_num++;
        }
      }
    }
    pc_full_res.points[full_num] = m_raw_pts_vec[i];
    pc_full_res.points[full_num].intensity = m_pts_info_vec[i].time_stamp;
    if (color_mode) {
      set_intensity(pc_full_res.points[full_num], i,
                    static_cast<E_intensity_type>(color_mode));
    }
    full_num++;
  }

  // printf("Get_features , corner num = %d, suface num = %d, blur from
  // %.2f~%.2f\r\n", corner_num, surface_num, minimum_blur, maximum_blur);
  pc_corners.resize(corner_num);
  pc_surface.resize(surface_num);
  pc_full_res.resize(full_num);
}

int main(int argc, char **argv) {
  // 初始化 ROS
  ros::init(argc, argv, "livox_feature_extractor");
  ros::NodeHandle nh;

  std::string cloudSource;
  ros::NodeHandle nh_private("~");
  nh_private.getParam("pcd_source", cloudSource);

  // mGridRes = nh_private.param("grid_res", 0.5);
  // mHeightTh = nh_private.param("height_th", 0.5);
  // mGetMinZFrames = nh_private.param("get_min_z_frames", 10);

  ROS_INFO("Get pcd cloud source msg: %s", cloudSource.c_str());
  ros::Subscriber pc_sub = nh.subscribe(cloudSource, 10, pointCloudCallback);

  std::thread process(processThread);

  ros::spin();
  process.join();

  return 0;
}
