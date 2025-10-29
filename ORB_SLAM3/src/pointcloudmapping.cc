ptr/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <pcl/io/pcd_io.h>

#include <boost/make_shared.hpp>

PointCloudMapping::PointCloudMapping(double resolution_)
{
    // 将传入的分辨率参数resolution_赋值给类的成员变量resolution
    this->resolution = resolution_;
    
    // 设置体素滤波器的叶子尺寸（三个维度使用相同的分辨率）
    voxel.setLeafSize( resolution, resolution, resolution);
    
    // 使用boost库创建共享指针管理的空点云对象，赋值给globalMap
    globalMap = boost::make_shared< PointCloud >( );

    // 创建并启动可视化线程，绑定到PointCloudMapping::viewer成员函数
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        // 获取shutDownMutex锁，保证线程安全
        unique_lock<mutex> lck(shutDownMutex);
        // 设置关闭标志位为true
        shutDownFlag = true;
        // 通知等待在keyFrameUpdated条件变量上的线程
        keyFrameUpdated.notify_one();
    }
    // 等待可视化线程结束
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    // 打印接收到的关键帧ID信息
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    
    // 获取keyframeMutex锁，保证线程安全
    unique_lock<mutex> lck(keyframeMutex);
    
    // 将关键帧指针存入keyframes向量
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );//这里存储的色彩信息
    depthImgs.push_back( depth.clone() );

    // 通知等待在keyFrameUpdated条件变量上的线程
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {//间隔三个像素点采样
            float d = depth.ptr<float>(m)[n];
            
            // 过滤无效深度值(小于1cm或大于10m)
            if (d < 0.01 || d>10)
                continue;
                
            // 创建点云点并计算3D坐标
            PointT p;
            p.z = d;  // Z坐标为深度值
            // 根据相机内参计算X,Y坐标(透视投影)
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;
            
            // 设置点的RGB颜色(从彩色图像获取)
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
            
            //可以在这里加入语义信息

            // 将点添加到临时点云
            tmp->points.push_back(p);
        }
    }

    // 获取关键帧的位姿(从世界坐标系到相机坐标系的变换)
    Eigen::Isometry3d T = ORB_SLAM3::Converter::toSE3Quat(kf->GetPose());
    
    // 创建最终点云对象
    PointCloud::Ptr cloud(new PointCloud);
    
    // 将点云从相机坐标系变换到世界坐标系
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    
    // 标记点云为非稠密(包含NaN或无效点)
    cloud->is_dense = false;

    // 输出生成的点云信息
    cout << "generate point cloud for kf " << kf->mnId 
         << ", size=" << cloud->points.size() << endl;
    
    return cloud;
}


void PointCloudMapping::viewer()
{
    // 创建PCL点云查看器
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        // 检查关闭标志位
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag) {
                break;  // 如果系统关闭，退出循环
            }
        }
        
        // 等待关键帧更新通知
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }

        // 获取当前关键帧数量
        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }

        // 处理新增的关键帧
        for (size_t i=lastKeyframeSize; i<N; i++)
        {
            // 为每个关键帧生成点云并合并到全局地图
            PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
            *globalMap += *p;
        }
        
        // 保存点云地图到文件
        pcl::io::savePCDFileBinary("vslam.pcd", *globalMap);
        
        // 对全局地图进行体素滤波降采样
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        globalMap->swap( *tmp );

        // 显示全局地图
        viewer.showCloud( globalMap );
        pcl::io::savePCDFileBinary("vslam.pcd", *globalMap);   //保存并输出地图
        cout << "show global map, size=" << globalMap->points.size() << endl;
        
        // 更新已处理的关键帧数量
        lastKeyframeSize = N;
    }
}

