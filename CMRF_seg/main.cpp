//
//  main.cpp
//  CGAL_rf
//
//  Created by Alive on 2020/12/7.
//  Copyright © 2020 Alive. All rights reserved.
//

#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:4244) // boost::number_distance::distance()
    // converts 64 to 32 bits integers
#endif

#include "algo_in_class.h"
#include "feature_and_cost.h"
#include "DFS_search.h"

int main (int argc, char** argv) {
    std::string root_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/";
    std::string file_name = "fused_minpxl2_2xsr_defParam_testArea";
    std::string file_path = root_path + file_name + ".ply";
    
//    std::string file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/fused_minpxl2_2xsr_defParam_testArea.ply";
    std::string filtered_file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias/featureBias_testArea.ply";
    std::string vector_file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/20230325_grid30/vector_file.txt";

    my_pts<pclPointRGB> pc(file_path);
    
    lower_outliers_removal<pclPointRGB>(pc.get_cloud());

    double voxel_reso = 0.5f, seed_reso = 2.0f;
    SV_clustering<pclPointRGB> sv_clustering(voxel_reso, seed_reso);
    SV_map<pclPointRGB> sv_clusters = sv_generation(pc.get_cloud(), sv_clustering, voxel_reso, seed_reso);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    PCT<pclPointLabel>::Ptr labeled_voxel_cloud = sv_clustering.getLabeledCloud();
    
    //Adjacency map
    std::multimap<std::uint32_t, std::uint32_t> sv_adjacency_map;
    sv_clustering.getSupervoxelAdjacency(sv_adjacency_map);
    int max_label = sv_clustering.getMaxLabel();

    std::vector<std::vector<pclPointLabel>> voxelized_labeled_points;
    voxelized_labeled_points.resize(max_label);
    for (const auto &p : labeled_voxel_cloud->points)
        if (p.label != 0)
            voxelized_labeled_points[p.label - 1].push_back(p);
    
    printf("Filtering minor voxels... \n");
    minor_voxel_filtering<pclPointRGB>(sv_clusters, sv_adjacency_map, max_label);

    std::vector<std::vector<double>> feature_vector;
    feature_vector.resize(max_label);

    //for grid elevation feature
    int grid_size = 30;
    std::vector<std::vector<double>> grids_height;
    std::vector<double> minMax;
    
    std::vector<double> elevation_real_val;

//    for (const auto &id : sv_adjacency_map)
//        std::cout <<id.first <<" " <<id.second <<"\n";

    //Feature extraction
    std::size_t feature_num = 6, label_type = 3;

    printf("Extracting features... \n\n");
    normalAngleFeature<pclPointRGB>(feature_vector, sv_clusters, max_label);
    gridHeightFeature<pclPointRGB>(feature_vector, sv_clusters, pc.get_cloud(),
                                   grids_height, minMax,
                                   elevation_real_val, max_label, grid_size);
    voxelPlanarityFeature<pclPointRGB>(feature_vector, sv_clusters,
                                       labeled_voxel_cloud, max_label);
    shapeCompactnessFeature<pclPointRGB>(feature_vector, sv_clusters,
                                         labeled_voxel_cloud, max_label);
    rgbColorFeature<pclPointRGB>(feature_vector, sv_clusters,
                                 labeled_voxel_cloud, pc.get_cloud(), max_label);
    neighborConvexConcaveFeature<pclPointRGB>(feature_vector, sv_clusters,
                                              sv_adjacency_map, max_label);
    
    PCT<pclPointLabel>::Ptr label_cloud(new PCT<pclPointLabel>);

    
    //Construct boost graph
    std::vector<vertex_descriptor> sv_vertices;
    std::map<std::uint32_t, std::size_t> sv_vertices_index_map;

    Graph sv_adjacency_graph; //Main field
    for (const auto &id : sv_clusters) {
        vertex_descriptor vd = boost::add_vertex(sv_adjacency_graph);
        sv_vertices.push_back(vd);
        
        sv_vertices_index_map.insert(std::make_pair(id.first, sv_vertices.size() - 1));
        sv_adjacency_graph[vd].vid = id.first;

        //cost initialization
        sv_adjacency_graph[vd].cost.resize(label_type);
        for (int label = 0; label < label_type; ++ label) {
            sv_adjacency_graph[vd].cost[label] = vertexCost(feature_vector[id.first - 1], label);
        }
        
        sv_adjacency_graph[vd].label = static_cast<int>(
                                                        std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end()))
                                                        );
    }
    
    //semantic correction
    std::size_t elevation_feature_index = 1;
    for (const auto &vd : sv_vertices) {
        std::uint32_t svid = sv_adjacency_graph[vd].vid;
        auto range = sv_adjacency_map.equal_range(svid);
        
        if (sv_adjacency_graph[vd].label == 0) {
            continue;
        } else if (sv_adjacency_graph[vd].label == 2) {
            double adjacent_roof_num = 0, adjacency_num = 0;
            for (auto it = range.first; it != range.second; ++ it) {
                adjacency_num ++;
                std::size_t vertex_index = sv_vertices_index_map.find(it->second)->second;
                
                if (vertex_index <= max_label) {
                    vertex_descriptor adj_vd = sv_vertices[vertex_index];
                    
                    //1st: Forbidden convex facade connection
                    if (sv_adjacency_graph[adj_vd].label == 1) {
                        pcl::Supervoxel<pclPointRGB>::Ptr
                            cv = sv_clusters.find(svid)->second,
                            av = sv_clusters.find(it->first)->second;
                        if (convexConcaveAdjacency<pclPointRGB>(cv, av) == 0) {
                            sv_adjacency_graph[vd].cost[2] = DBL_MAX;
//                            sv_adjacency_graph[vd].cost[2] *= 10;
                            break;
                            
                        }
                    } else if (sv_adjacency_graph[adj_vd].label == 0) {
                        adjacent_roof_num ++;
                    }
                }
            }
            
            //new 2nd: Forbidden overhalf roof connection
            if (adjacent_roof_num / adjacency_num >= 0.5)
                sv_adjacency_graph[vd].cost[2] = DBL_MAX;
            
            //3rd: limited height
            if (feature_vector[svid - 1][elevation_feature_index] >= 0.08) {
                sv_adjacency_graph[vd].cost[2] = DBL_MAX;
//                sv_adjacency_graph[vd].cost[2] *= 10;
            }
        } else { continue; }

    }
    
    //re-labeling
    for (const auto &vd : sv_vertices) {
        sv_adjacency_graph[vd].label = static_cast<int>(
                                                        std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end()))
                                                        );
        
        //pseudo-initialization
        for (const auto &p : voxelized_labeled_points[sv_adjacency_graph[vd].vid - 1]) {
            int ini_label = static_cast<int>(
                                             std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end())));
            pclPointLabel lp(p.x, p.y, p.z, ini_label);
            label_cloud->push_back(lp);
        }
    }
    
    std::string init_path = root_path + file_name + "_init.ply";
    pcl::PLYWriter writer;
    writer.write(init_path, *label_cloud, true);
    
    
    //pairwise connection
    for (const auto &edge : sv_adjacency_map) {
        if (edge.first == edge.second)
            continue;
        
        std::size_t first_vertex_index = sv_vertices_index_map.find(edge.first)->second,
                    second_vertex_index = sv_vertices_index_map.find(edge.second)->second;
        
        if (first_vertex_index <= max_label and second_vertex_index <= max_label) {
            if (!boost::edge(sv_vertices[first_vertex_index], sv_vertices[second_vertex_index], sv_adjacency_graph).second) {
                edge_descriptor ed = boost::add_edge(sv_vertices[first_vertex_index], sv_vertices[second_vertex_index], sv_adjacency_graph).first;
                
                //weight initialization
//                double feature_dist = std_vector_euclidean_dist(
//                    feature_vector[edge.first - 1],
//                    feature_vector[edge.second - 1]
//                );
//                std::cout <<feature_dist <<"\n";
                
                //original weight definition
//                double lcolor = sv_avg_color_value<pclPointRGB>(sv_clusters.find(edge.first)->second, pc.get_cloud()), rcolor = sv_avg_color_value<pclPointRGB>(sv_clusters.find(edge.second)->second, pc.get_cloud());
                double color_dist = abs(feature_vector[edge.first - 1][4] -
                                        feature_vector[edge.second - 1][4]);
                double spatial_dist = point_euclidean_dist(sv_clusters.find(edge.first)->second->centroid_,
                                         sv_clusters.find(edge.second)->second->centroid_) / seed_reso;
                double normal_dist = voxel_normal_dist<pclPointRGB>(sv_clusters.find(edge.first)->second,
                                                   sv_clusters.find(edge.second)->second);
                double feature_dist = color_dist * 0.02 + spatial_dist * 0.05 + normal_dist;
                
                
                //Logistic Projection
                sv_adjacency_graph[ed].weight = 1 - (1 / (1 + exp(-1 * (0.5 * feature_dist - 1))));
//                std::cout <<sv_adjacency_graph[ed].weight <<"\n\n";
                
            }
        }
    }
    
    //DFS test
//    vertex_iterator vBegin, vEnd;
//    for (boost::tie(vBegin, vEnd) = boost::vertices(sv_adjacency_graph); vBegin != vEnd; ++ vBegin) {
//        DFSVisitor vis;
//        std::cout <<sv_adjacency_graph[*vBegin].vid <<"\n";
//        boost::depth_first_search(sv_adjacency_graph, boost::visitor(vis));
//    }
    
    std::cerr << std::endl << "Alpha expansion..." << std::endl << std::endl;
    CGAL::alpha_expansion_graphcut(sv_adjacency_graph,
                                      get(&Edge_property::weight, sv_adjacency_graph),
                                      get(&Vertex_property::cost, sv_adjacency_graph),
                                      get(&Vertex_property::label, sv_adjacency_graph),
                                      CGAL::parameters::vertex_index_map(get(boost::vertex_index, sv_adjacency_graph)));
    

    PCT<pclPointLabel>::Ptr expansion_label_cloud(new PCT<pclPointLabel>),
                            expansion_ground_cloud(new PCT<pclPointLabel>);
    
    for (const auto &vd : sv_vertices) {
        for (auto &p : voxelized_labeled_points[sv_adjacency_graph[vd].vid - 1]) {
            pclPointLabel lp(p.x, p.y, p.z, sv_adjacency_graph[vd].label);
            expansion_label_cloud->push_back(lp);
            
            if (sv_adjacency_graph[vd].label == 2)
                expansion_ground_cloud->push_back(lp);
        }
    }
    
    std::string expand_path = root_path + file_name + "_expansion.ply",
                ground_path = root_path + file_name + "_ground.ply";
    pcl::PLYWriter expansion_writer;
    expansion_writer.write(expand_path, *expansion_label_cloud, true);
    expansion_writer.write(ground_path, *expansion_ground_cloud, true);
    
    
    return 0;
}

