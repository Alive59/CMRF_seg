//
//  option_def.h
//  CMRF_seg
//
//  Created by Alive on 2023/8/18.
//

#ifndef option_def_h
#define option_def_h

#include <ctime>
#include <cmath>
#include <vector>
#include <iostream>
#include <numeric>
#include <thread>

#include <CGAL/license/Classification.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>

#include <CGAL/boost/graph/alpha_expansion_graphcut.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/for_each.h>
#include <CGAL/Classification/Label_set.h>
#include <CGAL/property_map.h>
#include <CGAL/iterator.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/scalable_allocator.h>
#include <mutex>
#endif // CGAL_LINKED_WITH_TBB

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

//PCL
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>


#endif /* option_def_h */
