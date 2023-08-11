//
//  feature_and_cost.h
//  CGAL_algorithms
//
//  Created by Alive on 2023/7/24.
//  Copyright © 2023 Alive. All rights reserved.
//

#ifndef feature_and_cost_h
#define feature_and_cost_h

#include "algo_in_class.h"

/** \brief Transfer feature into vertex cost.
           label: roofs - 0; facades - 1; grounds - 2; clutters - 3 */
double vertexCost(std::vector<double> &feature,
                int label) {
    double cost_val = 0;
    
    switch (label) {
        case 0:
            cost_val = (1 - feature[0]) + (1 - feature[1]) + feature[2]
                        + (1 - feature[3]) + (1 - feature[4]) + (1 - feature[5]);
//                        + (1 - feature[6]);
            break;
            
        case 1:
            cost_val = feature[0] + feature[2] + feature[3] + (1 - feature[4]);
//                        + feature[6];
            break;
            
        case 2:
            cost_val = (1 - feature[0]) + feature[1] + feature[2]
                        + (1 - feature[3]) + feature[4] + feature[5];
//                        + (1 - feature[6]);
            break;
            
        case 3:
            cost_val = feature[1] + (1 - feature[2]) + feature[4];
            break;
            
        default:
            break;
    }
    
    return cost_val / 6.0;
}

/** \brief Normalize the computed feature value. */
void featureNormalization(std::vector<double> &feature_val,
                          double minVal, double maxVal) {
    std::vector<double> normalized_feature_val;
    for (double val : feature_val) {
        normalized_feature_val.push_back((val - minVal) / (maxVal - minVal));
    }
    
    feature_val = normalized_feature_val;
}

/** \brief Examine if the expansion continues to work.
           Return: the number of changed labels. */
int isExpansionWorking(std::vector<vertex_descriptor> &prev_sv_vertices,
                       std::vector<vertex_descriptor> &sv_vertices,
                       Graph &g, Graph &prev_g) {
    assert(prev_sv_vertices.size() == sv_vertices.size());
    
    int changed_num = 0;
    for (std::size_t idx = 0; idx < sv_vertices.size(); ++ idx) {
        if (prev_g[prev_sv_vertices[idx]].label != g[sv_vertices[idx]].label)
            changed_num ++;
    }
    
    return changed_num;
}

/** \brief Add new feature to a given feature vector for Graph-cut. */
void vertexFeatureEditor(std::vector<std::vector<double>> &feature_vector,
                         std::vector<double> &feature_value) {
    assert(feature_vector.size() == feature_value.size());
    
    for (std::size_t idx = 0; idx < feature_vector.size(); ++ idx)
        feature_vector[idx].push_back(feature_value[idx]);
}

/** \brief Calculate normal feature of a single voxel unit. */
template<typename pointT>
void normalAngleFeature(std::vector<std::vector<double>> &feature_vector,
                        SV_map<pointT> &clusters,
                        int max_label) {
    std::vector<double> val(max_label, 0);
    double val_min = 1, val_max = -1, feature_val = 0;
    
    for (const auto &cls : clusters) {
//        if (isnan(vector_horizon_angle(cls.second->normal_))) {
//            std::cout <<"nan: " <<cls.second->voxels_->size() <<"\n";
//        } else if (!vector_horizon_angle(cls.second->normal_)) {
//            std::cout <<"zero: " <<cls.second->voxels_->size() <<"\n";
//        }
        feature_val = abs(vector_horizon_angle(cls.second->normal_));
        val[cls.first - 1] = feature_val;
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
        
//        val.push_back(vector_horizon_angle(cls.second->normal_));
    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

/** \brief Calculate grid-height feature of a single voxel unit. */
template<typename pointT>
void gridHeightFeature(std::vector<std::vector<double>> &feature_vector,
                       SV_map<pointT> &clusters,
                       typename PCT<pointT>::Ptr &cloud,
                       std::vector<std::vector<double>> &grids,
                       std::vector<double> &minMax,
                       std::vector<double> &real_val,
                       int max_label,
                       int grid_size = 30) {
    std::vector<double> val(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    grid_division<pointT>(cloud, grids, minMax, grid_size);
    for (const auto &cls : clusters) {
        pclPoint centroid(cls.second->centroid_.x,
                          cls.second->centroid_.y,
                          cls.second->centroid_.z);
        
        int r_num = (centroid.y - minMax[2]) / grid_size,
            c_num = (centroid.x - minMax[0]) / grid_size;
        
        feature_val = centroid.z - grids[r_num][c_num];
        if (feature_val < 0)
            feature_val = 1e-6;
        
        val[cls.first - 1] = feature_val;
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
//        val.push_back(centroid.z - grids[r_num][c_num]);
    }
    
    //up-scale larger values
//    double scale_threshold = (val_max - val_min) * 0.03 + val_min,
//           scale_factor = 1.5;
//    for (const auto &cls : clusters) {
//        if (val[cls.first - 1] >= scale_threshold)
//            val[cls.first - 1] *= scale_factor;
//    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
    
    real_val = val;
}

/** \brief Calculate planarity feature of a single voxel unit. */
template<typename pointT>
void voxelPlanarityFeature(std::vector<std::vector<double>> &feature_vector,
                           SV_map<pointT> &clusters,
                           PCT<pclPointLabel>::Ptr &voxelized_labeled_cloud,
                           int max_label) {
    std::vector<double> val(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    std::vector<std::vector<pclPointLabel>> voxelized_labeled_points;
    voxelized_labeled_points.resize(max_label);
    for (const auto &p : voxelized_labeled_cloud->points) {
        if (!p.label) {
            continue;
        } else {
            voxelized_labeled_points[p.label - 1].push_back(p);
        }
    }
    
    for (const auto &cls : clusters) {
        Point_set eigen_pts;
        
        PCT<pclPoint>::Ptr eigen_cloud(new PCT<pclPoint>);
        eigen_cloud->push_back(pclPoint(cls.second->centroid_.x,
                                        cls.second->centroid_.y,
                                        cls.second->centroid_.z));
        for (const auto &p : voxelized_labeled_points[cls.first - 1])
            eigen_cloud->push_back(pclPoint(p.x, p.y, p.z));
        
        pcl_points_to_point_set<pclPoint>(eigen_cloud, eigen_pts);
        
        Neighborhood eigen_neighborhood(eigen_pts, eigen_pts.point_map());
        Local_eigen_analysis eigen = Local_eigen_analysis::create_from_point_set(
            eigen_pts,
            eigen_pts.point_map(),
            eigen_neighborhood.k_neighbor_query(static_cast<int>(cls.second->voxels_->size()))
        );
        
        std::array<float, 3> eigenvalue = eigen.eigenvalue(0);
        
        if (std::find(eigenvalue.begin(), eigenvalue.end(), 0) != eigenvalue.end()) {
            feature_val = 0;
        } else {
            feature_val = (eigenvalue[1] - eigenvalue[0])/ eigenvalue[2];
        }
        
        val[cls.first - 1] = feature_val;
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
        
//        val.push_back((eigenvalue[1] - eigenvalue[0])/ eigenvalue[2]);
    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

/** \brief Calculate shape compactness feature of a single voxel unit. */
template<typename pointT>
void shapeCompactnessFeature(std::vector<std::vector<double>> &feature_vector,
                             SV_map<pointT> &clusters,
                             PCT<pclPointLabel>::Ptr &voxelized_labeled_cloud,
                             int max_label) {
    std::vector<double> val(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    std::vector<std::vector<pclPointLabel>> voxelized_labeled_points;
    voxelized_labeled_points.resize(max_label);
    for (const auto &p : voxelized_labeled_cloud->points) {
        if (!p.label) {
            continue;
        } else {
            voxelized_labeled_points[p.label - 1].push_back(p);
        }
    }
    
    for (const auto &cls : clusters) {
        double shape_2d = 0, shape_3d = 0;
        std::vector<double> xy_minMax, z_value;
        cloud_xy_boundary<pointT>(cls.second->voxels_, xy_minMax);
        
        for (const auto &p : voxelized_labeled_points[cls.first - 1])
            z_value.push_back(p.z);
        
        std::sort(z_value.begin(), z_value.end());
        
        //original
        pclPoint p1(xy_minMax[0], xy_minMax[2], z_value[0]),
               p2(xy_minMax[1], xy_minMax[3], z_value[z_value.size() - 1]);
        shape_3d = point_euclidean_dist(p1, p2);
        
        //projected
        pclPoint p1p(xy_minMax[0], xy_minMax[2], 0),
               p2p(xy_minMax[1], xy_minMax[3], 0);
        shape_2d = point_euclidean_dist(p1p, p2p);
        
        feature_val = shape_2d / shape_3d;
        
        val[cls.first - 1] = feature_val;
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
//        val.push_back(shape_2d / shape_3d);
        
    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

/** \brief Calculate RGB feature of a single voxel unit. */
template<typename pointT>
void rgbColorFeature(std::vector<std::vector<double>> &feature_vector,
                     SV_map<pointT> &clusters,
                     PCT<pclPointLabel>::Ptr &voxelized_labeled_cloud,
                     typename PCT<pointT>::Ptr &original_cloud,
                     int max_label) {
    std::vector<double> val(max_label, 0), point_num(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    pcl::KdTreeFLANN<pointT> rgb_tree;
    int k = 1;
    std::vector<int> pointIdxKNNSearch(k);
    std::vector<float> pointKNNSquaredDistance(k);
    rgb_tree.setInputCloud(original_cloud);
    
    for (const auto &p : voxelized_labeled_cloud->points) {
        pointT searchPoint(p.x, p.y, p.z);
        
        if (!p.label)
            continue;
        
        if (rgb_tree.nearestKSearch(searchPoint, k, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
            val[p.label - 1] = (original_cloud->points[pointIdxKNNSearch[0]].r +
                                original_cloud->points[pointIdxKNNSearch[0]].g +
                                original_cloud->points[pointIdxKNNSearch[0]].b) / 3.0;
            point_num[p.label - 1] ++;
        }
    }
    
    for (const auto &cls : clusters) {
        val[cls.first] /= point_num[cls.first];
        feature_val = val[cls.first];
        
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
    }
    
//    for (const auto &cls : clusters) {
//        double voxel_rgb_grayscaled = 0;
//        std::size_t voxel_size = cls.second->voxels_->size();
//
//        //Integrate RGB using luminosity method
//        for (const auto &p : *cls.second->voxels_)
//            voxel_rgb_grayscaled += ((p.r + p.g + p.b) / 3.0) / voxel_size;
////            voxel_rgb_grayscaled += (p.r * 0.2989 + p.g * 0.5870 + p.b * 0.1140) / voxel_size;
//
//        feature_val = voxel_rgb_grayscaled;
//        val[cls.first - 1] = feature_val;
//        if (feature_val <= val_min) {
//            val_min = feature_val;
//        }
//        if (feature_val >= val_max) {
//            val_max = feature_val;
//        }
////        val.push_back(voxel_rgb_grayscaled);
//    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

/** \brief Calculate neighboring gradient sign feature of a single voxel unit. */
template<typename pointT>
void neighborConvexConcaveFeature(std::vector<std::vector<double>> &feature_vector,
                                 SV_map<pointT> &clusters,
                                 std::multimap<std::uint32_t, std::uint32_t> &sv_adjacency_map,
                                  int max_label) {
    //lccp convex-concave theory (Disabled for facade voxels and clutters)
    
    std::vector<double> val(max_label, 0);
    std::vector<double> convex_num(max_label, 0),
                     concave_num(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    for (auto adj = sv_adjacency_map.begin(); adj != sv_adjacency_map.end(); ++adj) {
        if (adj->first == adj->second or
            adj->first == 0 or
            adj->first > max_label or
            adj->second == 0 or
            adj->second > max_label)
            continue;
        
        typename SV_map<pointT>::iterator central_voxel = clusters.find(adj->first),
                                      adjacent_voxel = clusters.find(adj->second);
        if (central_voxel->first > max_label or adjacent_voxel->first > max_label)
            continue;
//        std::cout <<adj->first <<" " <<adj->second <<"\n";
        
//        pclPointRGB central_centroid = central_voxel->second->centroid_,
//                    adj_centroid = adjacent_voxel->second->centroid_;
//
//        //vector from centre to adja.
//        Eigen::Vector3d diff_vector({
//            adj_centroid.x - central_centroid.x,
//            adj_centroid.y - central_centroid.y,
//            adj_centroid.z - central_centroid.z
//        }),
//        central_normal = pcl_normal_to_eigen_vector(central_voxel->second->normal_),
//        adjacent_normal = pcl_normal_to_eigen_vector(adjacent_voxel->second->normal_);
//
//        double central_angle = eigen_vector3d_angle(central_normal, diff_vector),
//               adjacent_angle = eigen_vector3d_angle(adjacent_normal, diff_vector),
//               diff_angle = eigen_vector3d_angle(central_normal, adjacent_normal);
        
//        if (diff_angle > 10.0) {
//            if (central_angle > adjacent_angle) {
//                convex_num[adj->first - 1] ++;
//            } else {
//                concave_num[adj->first - 1] ++;
//            }
//        }
        if (convexConcaveAdjacency<pointT>(central_voxel->second, adjacent_voxel->second) == 0) {
            convex_num[adj->first - 1] ++;
        } else if (convexConcaveAdjacency<pointT>(central_voxel->second, adjacent_voxel->second) == 1){
            concave_num[adj->first - 1] ++;
        } else {}
    }
    
    for (const auto &cls : clusters) {
        feature_val = convex_num[cls.first - 1] - concave_num[cls.first - 1];
        val[cls.first - 1] = feature_val;
        if (feature_val <= val_min) {
            val_min = feature_val;
        }
        if (feature_val >= val_max) {
            val_max = feature_val;
        }
    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

/** \brief Calculate shape compactness feature of a single voxel unit. */
template<typename pointT>
void voxelIntensityFeature(std::vector<std::vector<double>> &feature_vector,
                           SV_map<pointT> &clusters,
                           PCT<pclPointLabel>::Ptr &voxelized_labeled_cloud,
                           int max_label) {
    std::vector<double> val(max_label, 0), point_num(max_label, 0);
    double val_min = DBL_MAX, val_max = DBL_MIN, feature_val = 0;
    
    std::vector<double> distance_sum(max_label, 0), distance_num(max_label, 0);
    std::vector<pclPoint> centroids;
    centroids.resize(max_label);
    
    for (const auto &cls : clusters)
        centroids[cls.first - 1] = pclPoint(cls.second->centroid_.x,
                                            cls.second->centroid_.y,
                                            cls.second->centroid_.z);
    
    for (const auto &p : voxelized_labeled_cloud->points) {
        if (!p.label)
            continue;
        
        distance_sum[p.label - 1] = sqrt(
                                         pow((p.x - centroids[p.label - 1].x), 2.0) +
                                         pow((p.y - centroids[p.label - 1].y), 2.0) +
                                         pow((p.z - centroids[p.label - 1].z), 2.0)
                                        );
        distance_num[p.label - 1] ++;
    }
    
    for (const auto &cls : clusters) {
        feature_val = distance_sum[cls.first - 1] / distance_num[cls.first - 1];
        val[cls.first - 1] = feature_val;
        
        if (distance_sum[cls.first - 1] <= val_min) {
            val_min = distance_sum[cls.first - 1];
        }
        if (distance_sum[cls.first - 1] >= val_max) {
            val_max = distance_sum[cls.first - 1];
        }
    }
    
    featureNormalization(val, val_min, val_max);
    vertexFeatureEditor(feature_vector, val);
}

#endif /* feature_and_cost_h */
