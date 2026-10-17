// Visual domain randomization for mujoco_ros2_control: recolour MJCF materials at runtime.
//
// Service /color_randomizer/set_colors (mujoco_dr_plugin/srv/SetColors): material names + RGBA.
// The request is validated in the service callback, queued, and applied by update() in the
// simulation thread; the reply is sent once it has been applied (or after a timeout, e.g. when
// the simulation is paused and update() isn't running), so frames rendered after the reply show
// the new colours.
//
// mujoco_ros2_control hands plugins a const mjModel*. mat_rgba is plain model memory that
// MuJoCo's visualizer reads on every scene update (mjv_updateScene), and nothing caches it, so
// writing it through a const_cast recolours every renderer that uses this model (the camera
// plugin and the viewer). Only mat_rgba is written - no sizes or pointers change.
//
// The topic name is absolute on purpose: relative names land inside the plugin's own node
// namespace (seen with the camera plugin, see ur5e_2f85_camera_mujoco.urdf.xacro).

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_dr_plugin/srv/set_colors.hpp"
#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_dr_plugin
{

class ColorRandomizerPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * /*data*/) override
  {
    node_ = node;
    model_ = const_cast<mjModel *>(model);  // see the file comment
    service_ = node_->create_service<srv::SetColors>(
      "/color_randomizer/set_colors",
      [this](const std::shared_ptr<srv::SetColors::Request> req,
             std::shared_ptr<srv::SetColors::Response> res) { on_set_colors(req, res); });
    RCLCPP_INFO(node_->get_logger(), "ColorRandomizerPlugin: %ld materials, service /color_randomizer/set_colors",
                static_cast<long>(model_->nmat));
    return true;
  }

  void update(const mjModel * /*model*/, mjData * /*data*/) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.empty()) {
      return;
    }
    for (const auto & change : pending_) {
      for (int k = 0; k < 4; ++k) {
        model_->mat_rgba[4 * change.mat_id + k] = change.rgba[k];
      }
    }
    pending_.clear();
    ++applied_generation_;
    applied_cv_.notify_all();
  }

  void cleanup() override
  {
    service_.reset();
    node_.reset();
  }

private:
  struct Change
  {
    int mat_id;
    float rgba[4];
  };

  void on_set_colors(const std::shared_ptr<srv::SetColors::Request> req,
                     std::shared_ptr<srv::SetColors::Response> res)
  {
    if (req->rgba.size() != 4 * req->materials.size()) {
      res->success = false;
      res->message = "rgba must hold 4 values per material";
      return;
    }
    std::vector<Change> changes;
    for (size_t i = 0; i < req->materials.size(); ++i) {
      const int id = mj_name2id(model_, mjOBJ_MATERIAL, req->materials[i].c_str());
      if (id < 0) {
        res->success = false;
        res->message = "unknown material '" + req->materials[i] + "'";
        return;
      }
      Change change{id, {}};
      for (int k = 0; k < 4; ++k) {
        const float v = req->rgba[4 * i + k];
        if (!(v >= 0.0f && v <= 1.0f)) {
          res->success = false;
          res->message = "rgba values must be in [0, 1]";
          return;
        }
        change.rgba[k] = v;
      }
      changes.push_back(change);
    }

    std::unique_lock<std::mutex> lock(mutex_);
    pending_.insert(pending_.end(), changes.begin(), changes.end());
    const uint64_t target = applied_generation_ + 1;
    const bool applied = applied_cv_.wait_for(lock, std::chrono::seconds(2),
                                              [&] { return applied_generation_ >= target; });
    res->success = applied;
    res->message = applied ? "applied " + std::to_string(changes.size()) + " material(s)"
                           : "queued but not applied within 2 s (simulation paused?)";
  }

  rclcpp::Node::SharedPtr node_;
  mjModel * model_ = nullptr;
  rclcpp::Service<srv::SetColors>::SharedPtr service_;
  std::mutex mutex_;
  std::condition_variable applied_cv_;
  std::vector<Change> pending_;
  uint64_t applied_generation_ = 0;
};

}  // namespace mujoco_dr_plugin

PLUGINLIB_EXPORT_CLASS(mujoco_dr_plugin::ColorRandomizerPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
