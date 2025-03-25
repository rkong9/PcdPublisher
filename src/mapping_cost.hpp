#ifndef _MAPPING_COST_HPP_
#define _MAPPING_COST_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>

using Point3D = Eigen::Vector3d;

// 定义 3D 直线的类型（用直线上一点和方向向量表示）
struct Line3D {
    Point3D point;  // 直线上的一点
    Point3D direction;  // 直线的方向向量（单位向量）
};

struct Plane3D {
    Point3D point;  // 直线上的一点
    Point3D direction;  // 直线的方向向量（单位向量）
};

struct Point2LineDis {
    Point2LineDis(const Point3D& point, const Line3D& line)
        : point_(point), line_(line) {}

    // 计算残差
    template <typename T>
    bool operator()(const T* const _q, const T* const _t, T* residual) const {
        // 这里优化的目标是获取从当前帧到上一帧的位姿变换
        // pose 是一个7维数组，前4个是位姿变化四元数，后3个是平移向量
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> angle_axis(pose);
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(pose + 3);

        Eigen::Quaternion<T> q{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t{_t[0], _t[1], _t[2]};
        Eigen::Matrix<T, 3, 1> pt_trans;
        // pt_trans = q * point_.cast<T>() + t;

        Eigen::Matrix<T, 3, 1> diff = pt_trans - line_.point.cast<T>();
        Eigen::Matrix<T, 3, 1> cross = diff.cross(line_.direction.cast<T>());

        // 将点从世界坐标系转换到相机坐标系
        // Eigen::Matrix<T, 3, 1> transformed_point =
        //    Eigen::AngleAxis<T>(angle_axis.norm(), angle_axis.normalized()) * point_.cast<T>() + translation;


        // 计算点到直线的距离
        residual[0] = cross.norm();

        return true;
    }

private:
    const Point3D point_;  // 3D 点
    const Line3D line_;    // 3D 直线
};

struct Point2PlaneDis {
    Point2PlaneDis(const Point3D& point, const Plane3D& plane)
        : point_(point), plane_(plane) {}

    template <typename T>
    bool operator()(const T* const _q, const T* const _t, T* residual) const {
        // pose 是一个7维数组，前4个是位姿变化四元数，后3个是平移向量
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> angle_axis(pose);
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(pose + 3);

        Eigen::Quaternion<T> q{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t{_t[0], _t[1], _t[2]};
        Eigen::Matrix<T, 3, 1> pt_trans;
        // pt_trans = q * point_.cast<T>() + t;

        Eigen::Matrix<T, 3, 1> diff = pt_trans - plane_.point.cast<T>();
        Eigen::Matrix<T, 3, 1> prj = diff.dot(plane_.direction.cast<T>());

        // 计算点到平面的距离
        residual[0] = prj.norm();

        return true;
    }

private:
    const Point3D point_;  // 3D 点
    const Plane3D plane_;    // 3D 直线
};

struct Point2LineDis_mb {
    Point2LineDis_mb(const Point3D& point, const Line3D& line, double motion_blur,
        Eigen::Quaterniond &q, Eigen::Vector3d &t)
        : point_(point), line_(line), motion_blur_(motion_blur), last_w_q(q), last_w_t(t) {
        }

    // 计算残差
    template <typename T>
    bool operator()(const T* const _q, const T* const _t, T* residual) const {
        // 这里优化的目标是获取从当前帧到上一帧的位姿变换
        // pose 是一个7维数组，前4个是位姿变化四元数，后3个是平移向量
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> angle_axis(pose);
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(pose + 3);
        Eigen::Quaternion<T> q_src((T)last_w_q.w(), (T)last_w_q.x(), (T)last_w_q.y(), (T)last_w_q.z());
        Eigen::Matrix<T, 3, 1> t_src((T)last_w_t.x(), (T)last_w_t.y(), (T)last_w_t.z());

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Quaternion<T> q_interpolate = Eigen::Quaternion<T>::Identity().slerp((T)motion_blur_, q_incre);
        Eigen::Matrix<T, 3, 1> t_interpolate = t_incre * T(motion_blur_);

        Eigen::Matrix<T, 3, 1> pt_ud;
        pt_ud = q_incre * point_.cast<T>() + t_incre;

        Eigen::Matrix<T, 3, 1> pt_trans = q_src * ( q_interpolate * pt_ud + t_interpolate ) + t_src;

        Eigen::Matrix<T, 3, 1> diff = pt_trans - line_.point.cast<T>();
        Eigen::Matrix<T, 3, 1> cross = diff.cross(line_.direction.cast<T>());

        // 计算点到直线的距离
        residual[0] = cross.norm();

        return true;
    }

private:
    const Point3D point_;  // 3D 点
    const Line3D line_;    // 3D 直线
    double motion_blur_;
    Eigen::Quaterniond last_w_q;
    Eigen::Vector3d last_w_t;
};

struct Point2PlaneDis_mb {
    Point2PlaneDis_mb(const Point3D& point, Eigen::Vector4d &plane, double motion_blur,
        Eigen::Quaterniond &q, Eigen::Vector3d &t)
        : point_(point), plane_(plane), motion_blur(motion_blur), last_w_q(q), last_w_t(t) {
        }

    template <typename T>
    bool operator()(const T* const _q, const T* const _t, T* residual) const {
        // pose 是一个7维数组，前4个是位姿变化四元数，后3个是平移向量
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> angle_axis(pose);
        // Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(pose + 3);
        Eigen::Quaternion<T> q_src((T)last_w_q.w(), (T)last_w_q.x(), (T)last_w_q.y(), (T)last_w_q.z());
        Eigen::Matrix<T, 3, 1> t_src((T)last_w_t.x(), (T)last_w_t.y(), (T)last_w_t.z());

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Quaternion<T> q_interpolate = Eigen::Quaternion<T>::Identity().slerp((T)motion_blur, q_incre);
        Eigen::Matrix<T, 3, 1> t_interpolate = t_incre * T(motion_blur);

        Eigen::Matrix<T, 3, 1> pt_ud;
        pt_ud = q_incre * point_.cast<T>() + t_incre;

        Eigen::Matrix<T, 3, 1> pt_trans = q_src * ( q_interpolate * pt_ud + t_interpolate ) + t_src;

        T diff = pt_trans(0, 0) * (T)plane_.x() + pt_trans(1, 0) * (T)plane_.y() + pt_trans(2, 0) * (T)plane_.z();
        Eigen::Matrix<T, 3, 1> dir((T)plane_[0], (T)plane_[1], (T)plane_[2]);

        // 计算点到平面的距离
        residual[0] = diff * diff / dir.dot(dir);

        return true;
    }

private:
    const Point3D point_;  // 3D 点
    Eigen::Vector4d plane_;
    double motion_blur;
    Eigen::Quaterniond last_w_q;
    Eigen::Vector3d last_w_t;
};

#endif

