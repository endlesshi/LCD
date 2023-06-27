#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>


#include "scancontext/Scancontext.h"

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

// Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
// Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

//mt params
int pcd_cnt=0;
float makeLCDTime,detLCDTime,totalLCDTime;
//mt params

std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

// PointCloudT::Ptr laserCloudFullRes(new PointCloudT);
// PointCloudT::Ptr laserCloudMapAfterPGO(new PointCloudT);

std::vector<PointCloudT::Ptr> keyframeLaserClouds; 
// std::vector<Pose6D> keyframePoses;
// std::vector<Pose6D> keyframePosesUpdated;
// std::vector<double> keyframeTimes;
// int recentIdxUpdated = 0;


pcl::VoxelGrid<PointT> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

// pcl::VoxelGrid<PointT> downSizeFilterICP;
// std::mutex mtxICP;
// std::mutex mtxPosegraph;
// std::mutex mtxRecentPose;

// PointCloudT::Ptr laserCloudMapPGO(new PointCloudT);
// pcl::VoxelGrid<PointT> downSizeFilterMapPGO;
// bool laserCloudMapPGORedraw = true;

// bool useGPS = true;
// // bool useGPS = false;
// sensor_msgs::NavSatFix::ConstPtr currGPS;
// bool hasGPSforThisKF = false;
// bool gpsOffsetInitialized = false; 
// double gpsAltitudeInitOffset = 0.0;
// double recentOptimizedX = 0.0;
// double recentOptimizedY = 0.0;

std::string save_directory;
// std::string pgKITTIformat, pgScansDirectory, pgSCDsDirectory;
// std::string odomKITTIformat;
// std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

//保存LSD的数据
std::ofstream fout_lcd_pair;
std::ofstream fout_lcd_time;

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");
    //cout<<"fileName="<<fileName<<endl;
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

void performSCLoopClosure(void)
{
    if( int(keyframeLaserClouds.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframeLaserClouds.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        mBuf.lock();
        fout_lcd_pair<<curr_node_idx<<' '<<prev_node_idx<<std::endl;
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    }
    // if( SCclosestHistoryFrameID == -1 ){
    //     cout<<pcd_cnt<<" not find LOOP CLOSURE."<<std::endl;
    // }
} // performSCLoopClosure

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
    TicToc t_lcd;
//make and save SCD
    //get PCD
    TicToc t_makeSCD;
    mBuf.lock();
    PointCloudT::Ptr thisKeyFrame(new PointCloudT);
    pcl::fromROSMsg(*_laserCloudFullRes, *thisKeyFrame);
    //fullResBuf.pop();
    mBuf.unlock();

    //downsample
    PointCloudT::Ptr thisKeyFrameDS(new PointCloudT);
    downSizeFilterScancontext.setInputCloud(thisKeyFrame);
    downSizeFilterScancontext.filter(*thisKeyFrameDS);

    mKF.lock(); 
    keyframeLaserClouds.push_back(thisKeyFrameDS);
    scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
    mKF.unlock();

    const auto& curr_scd = scManager.getConstRefRecentSCD();
    saveSCD(save_directory+"SCD/" + std::string(6-std::to_string(pcd_cnt).length(),'0')+std::to_string(pcd_cnt) + ".scd", curr_scd);
    //std::cout<<pcd_cnt<<" make SCD successful."<<endl;
    makeLCDTime=t_makeSCD.toc();

//detect LCD
    TicToc t_detSCD;
    performSCLoopClosure();
    detLCDTime=t_detSCD.toc();
    totalLCDTime=t_lcd.toc();
    fout_lcd_time<<std::to_string(pcd_cnt)<<" "<<std::to_string(makeLCDTime)<<" "<<std::to_string(detLCDTime)<<" "<<std::to_string(totalLCDTime)<<std::endl;

//reset params
    makeLCDTime=0.0;
    detLCDTime=0.0;
    totalLCDTime=0.0;
    pcd_cnt++;


} // laserCloudFullResHandler



int main(int argc, char **argv)
{
	ros::init(argc, argv, "lcd");
	ros::NodeHandle nh;

    // save directories 
	nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 

    fout_lcd_pair.open(save_directory+"LCD_pairs.txt");
    fout_lcd_time.open(save_directory+"LCD_time.txt");

    // pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    // pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);

    // system params 
	// nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	// nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    // keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    //downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    // double mapVizFilterSize;
	// nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    // downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudFullResHandler);

    // 发布PCD文件
    // ros::Rate loop_rate(100);  // 发布频率
    // PointCloudT::Ptr cloud(new PointCloudT);
    // while (ros::ok())
    // {
    //     ros::spinOnce();

    //     if (!fullResBuf.empty())
    //     {
    //         TicToc t_lcd;
    //         //get PCD
    //         mBuf.lock();
    //         PointCloudT::Ptr thisKeyFrame(new PointCloudT);
    //         pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
    //         fullResBuf.pop();
    //         mBuf.unlock();

    //         //downsample
    //         PointCloudT::Ptr thisKeyFrameDS(new PointCloudT);
    //         downSizeFilterScancontext.setInputCloud(thisKeyFrame);
    //         downSizeFilterScancontext.filter(*thisKeyFrameDS);

    //         mKF.lock(); 
    //         keyframeLaserClouds.push_back(thisKeyFrameDS);
    //         // keyframePoses.push_back(pose_curr);
    //         // keyframePosesUpdated.push_back(pose_curr); // init
    //         // keyframeTimes.push_back(timeLaserOdometry);
    //         //ROS_INFO("makeAndSaveScancontextAndKeys.");
    //         scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);


    //     }
   
    //     loop_rate.sleep();
    // }


 	ros::spin();

	return 0;
}