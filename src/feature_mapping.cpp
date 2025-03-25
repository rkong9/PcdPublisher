#include <cmath>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <condition_variable>
#include <mutex>
#include <future>
#include <thread>
#include <queue>
#include <pcl/segmentation/sac_segmentation.h>
#include "feature_mapping.hpp"
#include "mapping_cost.hpp"

FeatureMapping::FeatureMapping() {
    m_q_w_last = Eigen::Quaterniond(1, 0, 0, 0);
    m_t_w_last = Eigen::Vector3d::Zero();
}

FeatureMapping::~FeatureMapping() {

}

void FeatureMapping::readConfig(const YAML::Node &config) {
    mBufferSize = config["max_buffer_size"].as<int>();
    mSearchLinePts = config["search_line_pts"].as<int>();
    mValidLinePtsTh = config["valid_line_pts_th"].as<int>();
    mSearchLineAngleTh = config["search_line_angle_th"].as<float>();
    mLineHuberLossTh = config["line_huber_loss_th"].as<float>();
    mLineCheckRatio = config["line_check_ratio"].as<float>();

    mSearchSurfacePts = config["search_surface_pts"].as<int>();
    mValidSurfPtsTh = config["valid_surf_pts_th"].as<int>();
    mSearchSurAngleTh = config["search_surface_angle_th"].as<float>();
    mSurfHuberLossTh = config["surf_huber_loss_th"].as<float>();
    mSurfCheckRatio[0] = config["surf_check_ratio"][0].as<float>();
    mSurfCheckRatio[1] = config["surf_check_ratio"][1].as<float>();
}

void FeatureMapping::setInputData(
    PCloudXYZIPtr &pCorners, PCloudXYZIPtr &pSurfaces, PCloudXYZIPtr &pValidAll) {
    mCornersCurr = pCorners;
    mSurfaceCurr = pSurfaces;
    mValidAllCurr = pValidAll;

    PCloudXYZIPtr temp_corner(new PCloudXYZI);
    *temp_corner = *pCorners;
    mvHistoryCorner.push_back(temp_corner);
    if (mvHistoryCorner.size() > mBufferSize) {
        mvHistoryCorner.pop_front();
    }

    PCloudXYZIPtr temp_surface(new PCloudXYZI);
    *temp_surface = *pSurfaces;
    mvHistorySurface.push_back(temp_surface);
    if (mvHistorySurface.size() > mBufferSize) {
        mvHistorySurface.pop_front();
    }
}

void FeatureMapping::trans2End() {

}

Eigen::Matrix3d FeatureMapping::getCovMat(
    const PCloudXYZIPtr &pcloud, const std::vector<int> &indices) {

    std::vector<Eigen::Vector3d> vPts;
    Eigen::Vector3d center(Eigen::Vector3d::Zero());
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        auto &pt = pcloud->points[idx];
        Eigen::Vector3d ept(pt.x, pt.y, pt.z);
        center += ept;
        vPts.push_back(ept);
    }

    center /= indices.size();
    Eigen::Matrix3d covMat;
    for (auto &pt : vPts) {
        Eigen::Matrix<double, 3, 1> tmpZeroMean = pt - center;
        covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
    }
    return covMat;
}

inline Eigen::Vector3d FeatureMapping::getEPointFromP(const pcl::PointXYZI &pt) {
    return Eigen::Vector3d(pt.x, pt.y, pt.z);
}

int FeatureMapping::buildPoint2LineProblem(ceres::Problem &problem) {
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    ceres::LossFunction* loss_function = new ceres::HuberLoss(mLineHuberLossTh);
    std::vector<int> validLinePts;

    pcl::ModelCoefficients::Ptr pLineCoeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr pIndices(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> segLine;
    segLine.setModelType(pcl::SACMODEL_LINE);
    segLine.setMethodType(pcl::SAC_RANSAC);
    segLine.setMaxIterations(10);
    segLine.setOptimizeCoefficients(true);
    PCloudXYZIPtr pLineCloud(new PCloudXYZI);

    int block_nums(0);
    for (auto &pt : mCornersCurr->points) {
        int n(0);
        std::vector<int> validLines;
        {
            std::unique_lock lock(mCornerMtx);
            n = mCornersKdTree.nearestKSearch(pt, mSearchLinePts, pointIdxNKNSearch, pointNKNSquaredDistance);
        }

        float disTh = std::tan(mSearchLineAngleTh * 0.0174444) * pt.x;
        validLinePts.clear();
        for (int i = 0; i < n; i++) {
            if (pointNKNSquaredDistance[i] < disTh) {
                validLinePts.push_back(pointIdxNKNSearch[i]);
            }
        }

        if (validLinePts.size() < (size_t)mValidLinePtsTh) {
            ROS_WARN("get:%lu neighbour corner pts.(not enough:%d)", validLinePts.size(), mValidLinePtsTh);
            continue;
        }

        // add line valid check
        Eigen::Matrix3d covMat = getCovMat(mCornersAll, validLinePts);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

        if (saes.eigenvalues()[2] < mLineCheckRatio * saes.eigenvalues()[1]) {
            continue;
        }

        // add search result debug
        pLineCloud->clear();
        for (auto &i : validLinePts) {
            mSearchLineIndices.push_back(i);
            pLineCloud->push_back(mCornersCurr->points[i]);
        }

        // get line's parameters
        // Eigen::Vector3d linedir = saes.eigenvecs()[2];
        float th = 0.05 + (pt.x / 40.0) * 0.05;
        segLine.setDistanceThreshold(th);
        segLine.setInputCloud(pLineCloud);
        segLine.segment(*pIndices, *pLineCoeffs);

        if (pIndices->indices.size() < pLineCloud->size() / 2 ||
            pLineCoeffs->values.size() < 6) {
            continue;
        }

        // add line ceres block
        Line3D l3d;
        auto &v = pLineCoeffs->values;
        l3d.point = Eigen::Vector3d(v[0], v[1], v[2]);
        l3d.direction = Eigen::Vector3d(v[3], v[4], v[5]);
        double motion_blur = (pt.intensity - (int)pt.intensity) / 10000.0;
        ceres::CostFunction* costF = new ceres::AutoDiffCostFunction<Point2LineDis_mb, 1, 4, 3>(
            new Point2LineDis_mb(getEPointFromP(pt), l3d, motion_blur, m_q_w_last, m_t_w_last));

        problem.AddResidualBlock(costF, loss_function, m_para_buffer, m_para_buffer + 4);
        block_nums++;
    }
    std::unique(mSearchLineIndices.begin(), mSearchLineIndices.end());

    for (auto &i : mSearchLineIndices) {
        mSearchLineClouds->push_back(mCornersAll->points[i]);
    }
    ROS_INFO("add %d line blocks, %lu search points", block_nums, mSearchLineClouds->size());
    if (block_nums == 0) {
        ROS_WARN("no valid line blocks");
        return -1;
    }

    return 0;
}

int FeatureMapping::buildPoint2PlaneProblem(ceres::Problem &problem) {
    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    ceres::LossFunction* loss_function = new ceres::HuberLoss(mSurfHuberLossTh);
    std::vector<int> validSurfacesPts;

    pcl::ModelCoefficients::Ptr pPlaneCoeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr pIndices(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> segPlane;
    segPlane.setModelType(pcl::SACMODEL_PLANE);
    segPlane.setMethodType(pcl::SAC_RANSAC);
    segPlane.setMaxIterations(10);
    segPlane.setOptimizeCoefficients(true);
    PCloudXYZIPtr pPlaneCloud(new PCloudXYZI);

    int block_nums(0);
    for (auto &pt : mSurfaceCurr->points) {
        int n(0);
        {
            std::unique_lock lock(mCornerMtx);
            n = mSurfacesKdTree.nearestKSearch(pt, mSearchSurfacePts, pointIdxNKNSearch, pointNKNSquaredDistance);
        }

        float disTh = std::tan(mSearchLineAngleTh * 0.0174444) * pt.x;
        for (int i = 0; i < n; i++) {
            if (pointNKNSquaredDistance[i] < disTh) {
                validSurfacesPts.push_back(pointIdxNKNSearch[i]);
            }
        }

        if (validSurfacesPts.size() < (size_t)mValidSurfPtsTh) {
            continue;
        }

        // add plane valid check

        Eigen::Matrix3d covMat = getCovMat(mSurfaceAll, validSurfacesPts);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

        if (saes.eigenvalues()[2] > mSurfCheckRatio[0] * saes.eigenvalues()[1] ||
            saes.eigenvalues()[2] < mSurfCheckRatio[1] * saes.eigenvalues()[0]) {
            continue;
        }

        // add search result debug
        for (auto &i : validSurfacesPts) {
            mSearchSurfIndices.push_back(i);
            pPlaneCloud->push_back(mSurfaceAll->points[i]);
        }

        // get plane's parameters
        float th = 0.05 + (pt.x / 40.0) * 0.05;
        segPlane.setDistanceThreshold(th);
        segPlane.setInputCloud(pPlaneCloud);
        segPlane.segment(*pIndices, *pPlaneCoeffs);

        if (pIndices->indices.size() < pPlaneCloud->size() / 2 ||
            pPlaneCoeffs->values.size() < 4) {
            continue;
        }

        // add plane ceres block
        auto &v = pPlaneCoeffs->values;
        Eigen::Vector4d planeCoeffs(v[0], v[1], v[2], v[3]);
        double motion_blur = (pt.intensity - (int)pt.intensity) / 10000.0;
        ceres::CostFunction* costF = new ceres::AutoDiffCostFunction<Point2PlaneDis_mb, 1, 4, 3>(
            new Point2PlaneDis_mb(getEPointFromP(pt), planeCoeffs, motion_blur, m_q_w_last, m_t_w_last));

        problem.AddResidualBlock(costF, loss_function, m_para_buffer, m_para_buffer + 4);
        block_nums++;
    }

    std::unique(mSearchSurfIndices.begin(), mSearchSurfIndices.end());
    for (auto &i : mSearchSurfIndices) {
        mSearchSurClouds->push_back(mSurfaceAll->points[i]);
    }
    ROS_INFO("add %d surface blocks, %lu search points", block_nums, mSearchSurClouds->size());

    if (block_nums == 0) {
        ROS_WARN("no valid surface blocks");
        return -1;
    }

    return 0;
}

void FeatureMapping::assocateCloud2Map(PCloudXYZIPtr &pcloudIn) {
    for (auto &pt : pcloudIn->points) {
        Eigen::Vector3d ept(pt.x, pt.y, pt.z);
        double motion_blur = (pt.intensity - (int)pt.intensity) / 10000.0;
        Eigen::Quaterniond ud_q = Eigen::Quaterniond::Identity().slerp(1.0 - motion_blur, m_q_incre.inverse());
        Eigen::Vector3d ud_t = (1.0 - motion_blur) * (-m_t_incre);
        Eigen::Vector3d ud_pt = m_q_w_last * (ud_q * ept + ud_t) + m_t_w_last;
        pt.x = ud_pt.x();
        pt.y = ud_pt.y();
        pt.z = ud_pt.z();
    }
}

int FeatureMapping::calcTransform(Eigen::Quaterniond &q, Eigen::Vector3d &trans) {
    if (mvHistoryCorner.size() < mBufferSize || mvHistorySurface.size() < mBufferSize) {
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    PCloudXYZIPtr pCorner(new PCloudXYZI);
    for (auto &frame : mvHistoryCorner) {
        *pCorner += *frame;
    }

    mCornersAll = pCorner;
    mCornersKdTree.setInputCloud(mCornersAll);

    PCloudXYZIPtr pSurface(new PCloudXYZI);
    for (auto &frame : mvHistoryCorner) {
        *pSurface+= *frame;
    }
    mSurfaceAll = pSurface;
    mSurfacesKdTree.setInputCloud(mSurfaceAll);
    auto build_kd_tree = std::chrono::high_resolution_clock::now();

    // create Ceres problem
    ceres::Problem problem;
    int ret1 = buildPoint2LineProblem(problem);
    if (ret1 != 0) {
        ROS_WARN("build line block failed");
        return -1;
    }
    auto build_line_block = std::chrono::high_resolution_clock::now();

    int ret2 = buildPoint2PlaneProblem(problem);
    if (ret2 != 0) {
        ROS_WARN("build surface block failed");
        return -2;
    }
    auto build_surf_block = std::chrono::high_resolution_clock::now();

    // optimize

    // feat to map motion-blur
    // newscan, get l2w_q, l2w_q_last, incre_q_last(增量变换),

    // l2w_q = l2w_q * incre_q;

    // incre_q^ = incre_q_last
    // src_feat = incre_q^.slerp(blure).inverse() * cur_feat;
    // l2w_q = get_odo(src_feat, map);
    // === l2w_q = incre_q_c * l2w_q_last;
    // -> incre_q_c = l2w_q * l2w_q_last-1;

    // incre_q_c, src_feat =
    // std::cout << "get " << block << " blocks" << std::endl;
    // 配置求解器
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    // options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;

    // 求解
    ceres::Solve(options, &problem, &summary);
    auto solve_problem = std::chrono::high_resolution_clock::now();

    // 输出结果
    std::cout << summary.BriefReport() << std::endl;

    // trans and add curr frame to world
    // assocateCloud2Map(mValidAllCurr, m_q_w_last, m_t_w_last, m_q_incre, m_t_incre);
    assocateCloud2Map(mValidAllCurr);
    *mMap += *mValidAllCurr;
    pcl::VoxelGrid<pcl::PointXYZI> downSample;
    downSample.setLeafSize(0.05, 0.05, 0.05);
    downSample.setInputCloud(mMap);
    downSample.filter(*mMap);

    Eigen::Vector3d m_t_w_curr = m_q_w_last * m_t_incre + m_t_w_last;
    Eigen::Quaterniond m_q_w_curr = m_q_w_last * m_q_incre;

    auto build_kd_cost = std::chrono::duration_cast<std::chrono::milliseconds>(build_kd_tree - start_time);
    auto build_line_block_cost = std::chrono::duration_cast<std::chrono::milliseconds>(build_line_block - build_kd_tree);
    auto build_surf_block_cost = std::chrono::duration_cast<std::chrono::milliseconds>(build_surf_block - build_line_block);
    auto solve_cost = std::chrono::duration_cast<std::chrono::milliseconds>(solve_problem - build_surf_block);

    std::stringstream ss;
    ss << "kdTime:" << build_kd_cost.count() << "ms, line_block:" << build_line_block_cost.count() << "ms, "
       << "surf_block:" << build_surf_block_cost.count() << "ms, solve_problem:" << solve_cost.count() << "ms\n";
    ss << "curr move: " << (m_t_w_curr - m_t_w_last).transpose() << '\n';
    ROS_INFO("%s", ss.str().c_str());

    m_q_w_last = m_q_w_curr;
    m_t_w_last = m_t_w_curr;

    return 0;
}

void FeatureMapping::reset() {

}

void FeatureMapping::updateAllFeatures() {
    PCloudXYZIPtr pFeatCorners(new PCloudXYZI);
    PCloudXYZIPtr pFeatSurfaces(new PCloudXYZI);
    // for (auto &it = mvHistoryCorner.rbegin(); it != mvHistoryCorner.rend(); it++) {
    //     *pFeatCorners += *(*it);
    // }

    // {
    //   std::unique_lock lock(mCornerMtx);
    //   mCornersAll = pFeatCorners;
    //   mCornersKdTree.setInputCloud(mCornersAll);
    // }

    // for (auto &it = mvHistorySurface.rbegin(); it != mvHistorySurface.rend(); it++) {
    //     *pFeatSurfaces += *(*it);
    // }

    // {
    //     std::unique_lock lock(mSurfaceMtx);
    //     mSurfaceCurr = pFeatSurfaces;
    //     mSurfacesKdTree.setInputCloud(mSurfaceCurr);
    // }
}

std::map<int, std::shared_ptr<DataPack>> gMDataPack;
std::mutex gMutexBuff;
std::condition_variable gDataCv;

std::queue<std::shared_ptr<DataPack>> gQDataPack;

void lidarFeatureDataHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
    sensor_msgs::PointCloud2Ptr pMsg(new sensor_msgs::PointCloud2);
    *pMsg = *cloud_msg;
    std::string fid = cloud_msg->header.frame_id;

    std::unique_lock<std::mutex> lock(gMutexBuff);
    int seq = cloud_msg->header.seq;
    if (gMDataPack.find(seq) == gMDataPack.end()) {
        gMDataPack[seq] = std::make_shared<DataPack>();
    }

    if (fid == "c") {
        gMDataPack[seq]->pCorners = pMsg;
    } else if (fid == "s") {
        gMDataPack[seq]->pSurfaces = pMsg;
    } else if (fid == "f") {
        gMDataPack[seq]->pValidAll = pMsg;
    }

    if (gMDataPack[seq]->isComplete()) {
        gQDataPack.push(gMDataPack[seq]);
        gMDataPack.erase(seq);
    }
}

void processThread() {
    FeatureMapping fm;
    // fm->readConfig();
    std::shared_ptr<DataPack> pDataPack(nullptr);
    PCloudXYZIPtr pCorners(new PCloudXYZI);
    PCloudXYZIPtr pSurfaces(new PCloudXYZI);
    PCloudXYZIPtr pValidAll(new PCloudXYZI);
    Eigen::Quaterniond f2w_q(1, 0, 0, 0);
    Eigen::Vector3d f2w_t(0, 0, 0);

    while(ros::ok()) {
        // wait for data
        {
          std::unique_lock<std::mutex> lock(gMutexBuff);
          if (gQDataPack.size() == 0) {
            if (gDataCv.wait_for(lock, std::chrono::milliseconds(500)) ==
                std::cv_status::timeout) {
              ROS_WARN("wait pcd cloud overtime...");
              continue;
            }
          }
          pDataPack = gQDataPack.front();
          gQDataPack.pop();
        }

        pcl::fromROSMsg(*pDataPack->pCorners, *pCorners);
        pcl::fromROSMsg(*pDataPack->pSurfaces, *pSurfaces);
        pcl::fromROSMsg(*pDataPack->pValidAll, *pValidAll);
        fm.setInputData(pCorners, pSurfaces, pValidAll);
        int ret = fm.calcTransform(f2w_q, f2w_t);
        if (ret != 0) {
            if (ret > 0) {
                ROS_INFO("module initializing");
            } else {
                ROS_ERROR("compute trans failed!");
            }
            continue;
        }
        // get odom
            // fm->getOdom(int threadn);
        // assigne to map
        // pub msg
            // fm->getMap();
        // fm->getMap();
    }
}

int main(int argc, char **argv) {
  // 初始化 ROS
  ros::init(argc, argv, "feature_mapping");
  ros::NodeHandle nh;

  std::string cloudSource;
  ros::NodeHandle nh_private("~");
  nh_private.getParam("pcd_source", cloudSource);

  ros::Subscriber m_sub_laser_cloud_corner_last =
    nh.subscribe<sensor_msgs::PointCloud2>("/pc2_corners", 10000, lidarFeatureDataHandler);
  ros::Subscriber m_sub_laser_cloud_surf_last =
    nh.subscribe<sensor_msgs::PointCloud2>("/pc2_surface", 10000, lidarFeatureDataHandler);
  ros::Subscriber m_sub_laser_cloud_full_res =
    nh.subscribe<sensor_msgs::PointCloud2>("/pc2_full", 10000, lidarFeatureDataHandler);

  // ROS_INFO("Get pcd cloud source msg: %s", cloudSource.c_str());
  // ros::Subscriber pc_sub = nh.subscribe(cloudSource, 10, pointCloudCallback);

  std::thread process(processThread);

  ros::spin();
  process.join();

  return 0;
}
