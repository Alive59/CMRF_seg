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

int main (int argc, char** argv) {
    std::string file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/fused_minpxl2_2xsr_defParam_testArea.ply";
    std::string filtered_file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias/featureBias_testArea.ply";
    std::string vector_file_path = "/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/20230325_grid30/vector_file.txt";

    my_pts<pclPointRGB> pc(file_path);
    
    lower_outliers_removal<pclPointRGB>(pc.get_cloud());

//    double voxel_reso = 0.5f, seed_reso = 4.0f;
//    SV_clustering<pclPointRGB> sv_clustering(voxel_reso, seed_reso);
//    SV_map<pclPointRGB> sv_clusters = sv_generation(pc.get_cloud(), sv_clustering, voxel_reso, seed_reso);
    
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
//    voxelIntensityFeature<pclPointRGB>(feature_vector, sv_clusters,
//                                       labeled_voxel_cloud, max_label);
    

//    for (const auto &id : sv_adjacency_map)
//        std::cout <<id.first <<" " <<id.second <<"\n";
    
    //Feature values
//    for (unsigned int idx = 0; idx < feature_vector.size(); ++ idx) {
//        if (sv_clusters.find(idx + 1) != sv_clusters.end()) {
//            for (const auto &f : feature_vector[idx]) {
//                std::cout <<f <<" ";
//            }
//
//            std::cout <<"\n";
//        }
//    }
    
    PCT<pclPointLabel>::Ptr label_cloud(new PCT<pclPointLabel>);

    //Construct boost graph
    std::vector<vertex_descriptor> sv_vertices;
    std::map<std::uint32_t, std::size_t> sv_vertices_index_map;

    Graph sv_adjacency_graph;
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
        
        //temp_cost
//        double min_cost = *std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end());
//        for (double& val : sv_adjacency_graph[vd].cost) {
//            if (val == min_cost) {
//                val = 0;
//            } else {
//                val = 1;
//            }
//        }
        
        sv_adjacency_graph[vd].label = static_cast<int>(
                                                        std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end()))
                                                        );
        //pseudo-initialization
//        for (const auto &p : *id.second->voxels_) {
//            int ini_label = std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end()));
//            pclPointLabel lp(p.x, p.y, p.z, ini_label);
//            label_cloud->push_back(lp);
//        }
    }
    
    //semantic correction
    std::size_t elevation_feature_index = 1;
    for (const auto &vd : sv_vertices) {
        std::uint32_t svid = sv_adjacency_graph[vd].vid;
        auto range = sv_adjacency_map.equal_range(svid);
        
        if (sv_adjacency_graph[vd].label == 0) {
//            double adjacent_ground_num = 0, adjacency_num = 0;
//            for (auto it = range.first; it != range.second; ++ it) {
//                adjacency_num ++;
//                std::size_t vertex_index = sv_vertices_index_map.find(it->second)->second;
//
//                if (vertex_index <= max_label) {
//                    vertex_descriptor adj_vd = sv_vertices[vertex_index];
//
//                    if (sv_adjacency_graph[adj_vd].label == 2)
//                        adjacent_ground_num ++;
//                }
//            }
//
//            //1st: Forbidden overhalf ground connection
//            if (adjacent_ground_num / adjacency_num >= 0.5)
//                sv_adjacency_graph[vd].cost[0] = DBL_MAX;
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
//                        sv_adjacency_graph[vd].cost[2] = DBL_MAX;
                                            //2nd: Forbidden roof connection
//                        sv_adjacency_graph[vd].cost[2] *= 10;
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
        
        //cost-regularization
//        for (auto itr = sv_adjacency_graph[vd].cost.begin();
//                  itr != sv_adjacency_graph[vd].cost.end(); ++ itr) {
//            if (itr == std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end())) {
//                *itr = 0;
//            } else {
//                *itr = 1;
//            }
//        }
        
        //pseudo-initialization
        for (const auto &p : voxelized_labeled_points[sv_adjacency_graph[vd].vid - 1]) {
            int ini_label = static_cast<int>(
                                             std::distance(sv_adjacency_graph[vd].cost.begin(), std::min_element(sv_adjacency_graph[vd].cost.begin(), sv_adjacency_graph[vd].cost.end())));
            pclPointLabel lp(p.x, p.y, p.z, ini_label);
            label_cloud->push_back(lp);
        }
        
//        for (auto &element : sv_adjacency_graph[vd].cost)
//            element = 1 - (1 / (1 + exp(-1 * element)));
    }
    
    pcl::PLYWriter writer;
    writer.write("/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/fused_minpxl2_2xsr_defParam_testArea_init.ply", *label_cloud, true);
    
    
//    double max_color_gradient = maximum_color_gradient<pclPointRGB>(pc.get_cloud());
    for (const auto &edge : sv_adjacency_map) {
        if (edge.first == edge.second)
            continue;
        
        std::size_t first_vertex_index = sv_vertices_index_map.find(edge.first)->second,
                    second_vertex_index = sv_vertices_index_map.find(edge.second)->second;
        
        if (first_vertex_index <= max_label and second_vertex_index <= max_label) {
            if (!boost::edge(sv_vertices[first_vertex_index], sv_vertices[second_vertex_index], sv_adjacency_graph).second) {
                edge_descriptor ed = boost::add_edge(sv_vertices[first_vertex_index], sv_vertices[second_vertex_index], sv_adjacency_graph).first;
                
                //weight initialization
                double feature_dist = std_vector_euclidean_dist(
                    feature_vector[edge.first - 1],
                    feature_vector[edge.second - 1]
                );
                std::cout <<feature_dist <<"\n";
//                sv_adjacency_graph[ed].weight = feature_dist /
//                                                    static_cast<double>(feature_num);
                
                //Amplified discriminability weight
//                double feature_dist = 0, dist_value = 0;
//                for (std::size_t idx = 0; idx < feature_vector[edge.first - 1].size(); ++ idx) {
//                    if (idx == 0)
//                        dist_value = (feature_vector[edge.first - 1][idx] -
//                                      feature_vector[edge.second - 1][idx]) * 3;
//
//                    feature_dist += dist_value * dist_value;
//                }
                
//                feature_dist = feature_dist / static_cast<double>(feature_num);
//                sv_adjacency_graph[ed].weight = 1 - (1 / (1 + exp(-0.1 *  feature_dist)));
                sv_adjacency_graph[ed].weight = 1 - (1 / (1 + exp(-1 * (feature_dist - 1))));
                std::cout <<sv_adjacency_graph[ed].weight <<"\n\n";
                
//                sv_adjacency_graph[ed].weight = feature_dist / 2;
//                sv_adjacency_graph[ed].weight = 0.2;
                
                // =======
                
//                pcl::Supervoxel<pclPointRGB>::Ptr
//                    sv1 = sv_clusters.find(edge.first)->second,
//                    sv2 = sv_clusters.find(edge.second)->second;
//                double
//                    euclidean_dist = voxel_euclidean_dist<pclPointRGB>(sv1, sv2,
//                                                                       seed_reso),
//                    color_dist = voxel_color_dist<pclPointRGB>(sv1, sv2) /
//                                                               max_color_gradient,
//                    normal_dist = voxel_normal_dist<pclPointRGB>(sv1, sv2);

//                sv_adjacency_graph[ed].weight = (euclidean_dist * 0.2 +
//                                                color_dist * 0.8 + normal_dist * 2)
//                                                / 3;
//                sv_adjacency_graph[ed].weight = normal_dist * 1.0 +
//                                                euclidean_dist * 0.1 +
//                                                color_dist * 0.4;
//                sv_adjacency_graph[ed].weight = 1;
//                sv_adjacency_graph[ed].weight = normal_dist;
            }
        }
        
    }
    
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
    
    pcl::PLYWriter expansion_writer;
    expansion_writer.write("/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/fused_minpxl2_2xsr_defParam_testArea_expansion.ply", *expansion_label_cloud, true);
    expansion_writer.write("/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias_2xsr/fused_minpxl2_2xsr_defParam_testArea_ground.ply", *expansion_ground_cloud, true);
    
    //Visualization
//    for (auto itr = cls.begin(); itr != cls.end(); ++ itr) {
//        for (auto sv_itr = itr->second->voxels_->begin(); sv_itr != itr->second->voxels_->end(); ++ sv_itr) {
//            pclPointRGB p = *sv_itr;
//            p.r = ((itr->second->voxels_->size() + 1) * 32) % 200;
//            p.g = ((itr->second->voxels_->size() + 1) * 64) % 200;
//            p.b = ((itr->second->voxels_->size() + 1) * 128) % 200;
//            scatter_filtered_cloud->push_back(p);
//        }
//
//    }

    //Extract centroids for vector data
//    std::ofstream vector_file(vector_file_path, std::ios::trunc);
//
//    for (auto p : scatter_filtered_cloud->points) {
//        vector_file << p.x <<" ";
//        vector_file << p.y <<" ";
//        vector_file << p.z <<" ";
//        vector_file << std::endl;
//    }


    //Write the output
//    pcl::PLYWriter writer;
//    writer.write("/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias/featureBias_sor_filtered.ply", *(pc_with_sv.get_cloud()));

    // ========================================================================
    
    //Alpha-expansion
    std::array<char, 3> labels = { 'O', 'X', 'o' };
    
    std::array<std::array<int, 6>, 5> input = { { { 0, 2, 0, 1, 1, 1 },
                                                  { 0, 0, 1, 0, 1, 2 },
                                                  { 2, 0, 1, 1, 2, 2 },
                                                  { 0, 1, 1, 2, 2, 0 },
                                                  { 1, 1, 2, 0, 2, 2 } } };

    std::array<std::array<vertex_descriptor, 6>, 5> vertices;

    // Init vertices from values
    Graph g;
    for (std::size_t i = 0; i < input.size(); ++ i) {
        for (std::size_t j = 0; j < input[i].size(); ++ j) {
            vertices[i][j] = boost::add_vertex(g);
            g[vertices[i][j]].label = input[i][j];

            g[vertices[i][j]].cost.resize(3, 1); //Each represents a label
            // Cost of assigning this vertex to any label is positive except
            // for current label whose cost is 0 (favor init solution)
            g[vertices[i][j]].cost[std::size_t(input[i][j])] = 0;
        }
    }

//    g[vertices[1][3]].cost[1] = DBL_MAX;
//    g[vertices[1][3]].cost[2] = DBL_MAX;

    // Display input values
    std::cerr << "Input:" << std::endl;
    for (std::size_t i = 0; i < vertices.size(); ++ i) {
        for (std::size_t j = 0; j < vertices[i].size(); ++ j) {
            std::cerr << labels[std::size_t(g[vertices[i][j]].label)];
        }
        std::cerr << std::endl;
    }

    // Init adjacency
    double weight = 0.5;
    for (std::size_t i = 0; i < vertices.size(); ++ i) {
        for (std::size_t j = 0; j < vertices[i].size(); ++ j) {
          // Neighbor vertices are connected
            if (i < vertices.size() - 1) {
                edge_descriptor ed = boost::add_edge (vertices[i][j], vertices[i+1][j], g).first;

                g[ed].weight = weight;
            }

            if (j < vertices[i].size() - 1) {
                edge_descriptor ed = boost::add_edge (vertices[i][j], vertices[i][j+1], g).first;

                g[ed].weight = weight;
            }
        }
    }

    std::cerr << std::endl << "Alpha expansion..." << std::endl << std::endl;

    CGAL::alpha_expansion_graphcut (g,
                                      get (&Edge_property::weight, g),
                                      get (&Vertex_property::cost, g),
                                      get (&Vertex_property::label, g),
                                      CGAL::parameters::vertex_index_map (get (boost::vertex_index, g)));

    // Display output graph
    std::cerr << "Output:" << std::endl;
    for (std::size_t i = 0; i < vertices.size(); ++ i) {
        for (std::size_t j = 0; j < vertices[i].size(); ++ j)
          std::cerr << labels[std::size_t(g[vertices[i][j]].label)];
        std::cerr << std::endl;
    }

    
    return 0;
}

