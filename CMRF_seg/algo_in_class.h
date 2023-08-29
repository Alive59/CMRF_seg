//
//  algo_in_class.h
//  CGAL_algorithms
//
//  Created by Alive on 2022/11/11.
//  Copyright © 2022 Alive. All rights reserved.
//

#ifndef algo_in_class_h
#define algo_in_class_h

#include "option_def.h"

//CGAL
namespace Classification = CGAL::Classification;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Point_set_3<Point> Point_set;
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3;
typedef Point_set::Point_map Pmap;
typedef Point_set::Property_map<float> Imap;
typedef Point_set::Property_map<unsigned char> UCmap;
typedef Point_set::Vector_map Vector_map;

typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set
    Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Point_set_neighborhood<Kernel, Point_set, Pmap>       Neighborhood;
typedef Classification::Local_eigen_analysis                                    Local_eigen_analysis;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;
typedef Classification::ETHZ::Random_forest_classifier
    Classifier;


//PCL
template <typename pointT>
using PCT = pcl::PointCloud<pointT>;

template <typename pointT>
using SV_clustering = pcl::SupervoxelClustering<pointT>;

template <typename pointT>
using SV_map = std::map <std::uint32_t, typename pcl::Supervoxel<pointT>::Ptr>;

typedef pcl::PointXYZ pclPoint;
typedef pcl::PointXYZRGBA pclPointRGB;
typedef pcl::PointNormal pclPointN;
typedef pcl::PointXYZL pclPointLabel;
typedef pcl::PointXYZRGBL pclPointRGBLabel;
typedef pcl::Normal pclN;


//Boost
struct Vertex_property
{
    std::uint32_t vid;
    int label;
    std::vector<double> cost;
};
struct Edge_property
{
    double weight;
};

using Graph = boost::adjacency_list <boost::setS,
                                     boost::vecS,
                                     boost::undirectedS,
                                     Vertex_property,
                                     Edge_property>;
using GT = boost::graph_traits<Graph>;
using vertex_descriptor = GT::vertex_descriptor;
using edge_descriptor = GT::edge_descriptor;

using vertex_iterator = GT::vertex_iterator;


void pts_read(std::string pts_path, Point_set &pts);
template <typename pointT> void pcl_read(std::string pcl_path, typename PCT<pointT>::Ptr &cloud);
template <typename pointT> void point_set_to_pcl_points (const Point_set& pRange, typename PCT<pointT>::Ptr& cloud);
template <typename pointT> void pcl_points_to_point_set (const typename PCT<pointT>::Ptr& cloud, Point_set& pRange);
template <typename pointT> void pc_resolution_sampling(typename PCT<pointT>::Ptr &cloud, std::vector<float> &reso, int sample_num = 200);
template <typename pointT> SV_map<pointT> sv_generation(typename PCT<pointT>::Ptr &cloud, SV_clustering<pointT> &s, float voxel_reso, float seed_reso);
double vector_euclidean_dist(Vector v1, Vector v2);
template<typename pointT> double point_euclidean_dist(pointT &p1, pointT &p2);
template <typename elementT> std::vector<std::size_t> interquartile_outlier_detection (typename std::vector<elementT>& vec);
template <typename pointT> void pyramid_hierarchical_constraint(typename std::vector<std::vector<pointT>>& cloud_raster_dem, std::vector<std::vector<pointT>>& cloud_raster, int row, int col, double z_thres);
double eigen_vector3d_angle(Eigen::Vector3d &v1, Eigen::Vector3d &v2);
template<typename normalT> double vector_horizon_angle(normalT& n);
template<typename normalT> Eigen::Vector3d pcl_normal_to_eigen_vector(normalT &n);


template <typename pointT>
class my_pc {
protected:
    Point_set pts;
    typename PCT<pointT>::Ptr cloud;
    
public:
    // t is the identifier for input type, 1 for PCL, 2 for CGAL
    my_pc(std::string path, int t = 1) {
        PCT<pointT> shared_cloud;
        cloud = shared_cloud.makeShared();
        
        if (t == 1) {
            pcl_read<pointT>(path, cloud);
        } else if (t == 2) {
            pts_read(path, pts);
        } else {
            std::cout <<"Invalid identifiers!" <<std::endl;
            exit(-1);
        }
    }
    my_pc(typename PCT<pointT>::Ptr pc) {
        cloud = pc;
    }
    my_pc() {}
    
    Point_set& getPts() { return pts; }
    typename PCT<pointT>::Ptr& get_cloud() { return cloud; }
    void setPts(Point_set& p) { pts = p; }
    void set_cloud(typename PCT<pointT>::Ptr &cl) { cloud = cl; }
    void pc_transformer() {
        if (pts.size() == 0) {
            pcl_points_to_point_set<pointT>(cloud, pts);
        } else if (cloud->size() == 0){
            point_set_to_pcl_points<pointT>(pts, cloud);
        } else {
            return;
        }
    }
    ~my_pc() {
        std::cout <<"Releasing Ptrs..." <<std::endl;
        cloud.reset();
    }
};

template <typename pointT>
class my_pts : public my_pc<pointT> {
    Imap label_map;
    UCmap red_map, green_map, blue_map;
    
    /** \brief read the stored label map with training data*/
    void label_map_read(std::string map_name) {
        bool found = false;
        boost::tie(label_map, found) = this->getPts().template property_map<float>(map_name);
        if (!found)
            boost::tie(label_map, found) = this->getPts().template add_property_map<float>(map_name, -1.);
    }
    
    /** \brief read the stored RGB color map with training data*/
    void color_map_read() {
        bool red_found = false, green_found = false, blue_found = false;
        boost::tie(red_map, red_found) = this->getPts().template property_map<unsigned char>("red");
        boost::tie(green_map, green_found) = this->getPts().template property_map<unsigned char>("green");
        boost::tie(blue_map, blue_found) = this->getPts().template property_map<unsigned char>("blue");
    }
    
public:
    my_pts(std::string path, int t = 1, std::string map_name = "") : my_pc<pointT>(path, t) {
        if (!map_name.empty())
            label_map_read(map_name);
        
        color_map_read();
    }
    my_pts(typename PCT<pointT>::Ptr pc, std::string map_name = "") : my_pc<pointT>(pc){
        if (!map_name.empty())
            label_map_read(map_name);
        
        color_map_read();
    }
    my_pts() : my_pc<pointT>() {}
    
    /** \brief return the stored label map with training data*/
    Imap get_label_map() { return label_map; }
    /** \brief return the stored RGB color map array with training data*/
    UCmap * get_color_map() {
        static UCmap map[3] = {red_map, green_map, blue_map};
        return map;
    }
    
    ~my_pts() { }
};

template <typename pointT>
class my_pc_with_sv : public my_pts<pointT> {
    SV_clustering<pointT> sv_clustering;
    SV_map<pointT> clusters;
    
public:
    my_pc_with_sv(std::string path, float voxel_reso = 0.0f, float seed_reso = 0.0f, int t = 1, std::string map_name = "") : my_pts<pointT>(path, t, map_name), sv_clustering(0.0f, 0.0f)  {
        supervoxel_generation(voxel_reso, seed_reso);
    }
    my_pc_with_sv(SV_clustering<pointT> sc, SV_map<pointT> c, typename PCT<pointT>::Ptr pc, std::string map_name = "") : my_pts<pointT>(pc, map_name), sv_clustering(0.0f, 0.0f) {
        sv_clustering = sc;
        clusters = c;
    }
    my_pc_with_sv(typename PCT<pointT>::Ptr &pc, float voxel_reso, float seed_reso, std::string map_name = "") : my_pts<pointT>(pc, map_name), sv_clustering(0.0f, 0.0f) {
        supervoxel_generation(voxel_reso, seed_reso);
    }
    
    
    void supervoxel_generation(float voxel_reso, float seed_reso) {
        clusters = sv_generation<pointT>(this->cloud, sv_clustering, voxel_reso, seed_reso);
    }
    SV_map<pointT>& get_clusters() { return clusters; }
    SV_clustering<pointT>& get_clustering() { return sv_clustering; }
    
    ~my_pc_with_sv() {}
};

template <typename pointT>
class my_eigen : public my_pc_with_sv<pointT> {
    std::vector<std::vector<double>> eigen_value_list;
    
public:
    my_eigen(std::string path, int t = 1, std::string map_name = "") : my_pc_with_sv<pointT>(path, t, map_name) {
        if (t == 1) {
            eigen_value_list.resize(this->cloud->size());
        } else if (t == 2) {
            eigen_value_list.resize(this->pts.size());
        } else {
            exit(0);
        }
    }
    my_eigen(SV_clustering<pointT> sc, SV_map<pointT> c, typename PCT<pointT>::Ptr pc, std::string map_name = "") : my_pc_with_sv<pointT>(sc, c, pc, map_name) {
        eigen_value_list.resize(this->cloud->size());
    }
    
    my_eigen() : my_pts<pointT>() {}
    
    std::vector<std::vector<double>>& get_list() { return eigen_value_list; }
    void push_to_list(std::vector<double>& value) { eigen_value_list.push_back(value); }
    
    ~my_eigen() { }
};

template <typename pointT>
class my_saliency : public my_pts<pointT> {
    Vector_map normal_map;
    Vector dom_vector;
    double * saliency_value;
    double * cluster_probability;
    long * cluster_idx;
    double geometric_centroid;
    
    /** \brief create a normal map for original point cloud*/
    void create_normal_map(Point_set& pts) {
        if (!(pts.has_normal_map()))
            pts.add_normal_map();
    }
    
public:
    my_saliency(std::string pts_path, std::string map_name = "") : my_pts<pointT>(pts_path, 1, map_name) {
        create_normal_map(this->pts);
        saliency_value = new double[this->pts.size()];
        cluster_idx = new long[this->pts.size()];
        cluster_probability = new double[9];
        geometric_centroid = 0.0;
    }

    double hyperbolic_tangent_space_projection(double& dist) {
        return 0.5 + tanh(2 * (dist - geometric_centroid));
    }
    
    /** \brief K-means clustering of normal vectors*/
    void kmeans_clustering(int epochs, int k = 8) {
        std::vector<Vector> normals;
        for (size_t idx = 0; idx < this->pts.size(); ++ idx) {
            normals.push_back(Vector(normal_map[idx].x(), normal_map[idx].y(), normal_map[idx].z()));
        }
        
        std::vector<Vector> centroids;
        double * minDist = new double[normals.size()];
        long * cluster = new long[normals.size()];
        for (int i = 0; i < normals.size(); ++ i) {
            minDist[i] = __DBL_MAX__;
            cluster[i] = -1;
        }
        
        srand((unsigned int)time(0)); // random seed
        size_t n = normals.size();
        for (int i = 0; i < k; ++ i) {
            centroids.push_back(normals[rand() % n]);
        }
        
        std::vector<int> nPoints;
        for (int e = 0; e < epochs; ++e) {
            nPoints.clear();
            std::vector<double> sumX, sumY, sumZ;
            for (int i = 0; i < k; ++ i) {
                nPoints.push_back(0);
                sumX.push_back(0.0);
                sumY.push_back(0.0);
                sumZ.push_back(0.0);
            }
            
            for (std::vector<Vector>::iterator itr = begin(centroids);
                 itr != end(centroids); ++ itr) {
                long clusterID = itr - begin(centroids);
                
                int idx = 0;
                for (std::vector<Vector>::iterator i = normals.begin();
                     i != normals.end(); ++ i) {
                    double dist = vector_euclidean_dist(*itr, *i);
                    if (dist < minDist[idx]) {
                        minDist[idx] = dist;
                        cluster[idx] = clusterID;
                    }
                    
                    idx ++;
                }
                
            }
            
            if (e == epochs)
                break;
            
            int idx = 0;
            for (std::vector<Vector>::iterator itr = normals.begin();
                 itr != normals.end(); ++ itr) {
                nPoints[cluster[idx]] += 1;
                sumX[cluster[idx]] += itr->x();
                sumY[cluster[idx]] += itr->y();
                sumZ[cluster[idx]] += itr->z();
                
                minDist[idx] = __DBL_MAX__;
                cluster[idx] = -1;
                idx ++;
            }
            
            for (size_t i = 0; i < centroids.size(); ++ i) {
                centroids[i] = Vector(sumX[i] / nPoints[i], sumY[i] / nPoints[i], sumZ[i] / nPoints[i]);
            }
        }
        
        set_cluster_idx(*cluster);
        //get dominant vector
        int max_num = 0, max_idx = -1;
        for (int i = 0; i < k; ++ i) {
            if (nPoints[i] > max_num) {
                max_num = nPoints[i];
                max_idx = i;
            }
        }
        
        set_dom_vector(centroids[max_idx]);
    }

    
    void geometric_centroid_calculation() {
        double * h_k = new double[9];
        for (size_t i = 0; i < this->pts.size(); ++ i) {
            double dist = vector_euclidean_dist(dom_vector, normal_map[i]);
            saliency_value[i] = dist;
            h_k[cluster_idx[i]] += dist;
            cluster_probability[cluster_idx[i]] ++;
        }
        
        for (int idx = 0; idx < 8; ++ idx) {
            h_k[idx] = h_k[idx] / cluster_probability[idx];
            cluster_probability[idx] /= this->pts.size();
        }
        
        double upper_sum = 0.0, lower_sum = 0.0;
        for (int i = 0; i < 8; ++ i) {
            upper_sum += log(1 / cluster_probability[i]) * h_k[i];
            lower_sum += log(1 / cluster_probability[i]);
        }
        
        geometric_centroid = upper_sum / lower_sum;
    }
    
    void set_saliency_value(int& idx, double& dist) {
        double h_dist = hyperbolic_tangent_space_projection(dist);
        if (h_dist > hyperbolic_tangent_space_projection(geometric_centroid)) {
            saliency_value[idx] = 1;
        }else{
            saliency_value[idx] = 0;
        }
    }
    
    Vector_map& get_normal_map() { return normal_map; }
    void set_dom_vector(Vector& v) { dom_vector = v; }
    void set_cluster_prob(int& idx, double& p) { cluster_probability[idx] = p; }
    void set_cluster_idx(long& cluster) { cluster_idx = &cluster; }
    ~my_saliency() { delete[] saliency_value; }
};

// ================================================================================

void pts_read(std::string pts_path, Point_set &pts) {
    std::ifstream pin (pts_path.c_str(), std::ios::binary);
    pin >> pts;
}

template<typename pointT>
void pcl_read(std::string pcl_path, typename PCT<pointT>::Ptr &cloud) {
    pcl::PLYReader reader;
    reader.read(pcl_path, *cloud);
}

template<typename pointT>
void point_set_to_pcl_points (const Point_set& pRange, typename PCT<pointT>::Ptr& cloud) {
    cloud->points.resize(pRange.size());
    for (std::size_t i = 0; i < pRange.size(); i ++) {
        Point temp = pRange.point(i);
        cloud->points[i].x = temp.x();
        cloud->points[i].y = temp.y();
        cloud->points[i].z = temp.z();
    }
}

template<typename pointT>
void pcl_points_to_point_set (const typename PCT<pointT>::Ptr& cloud, Point_set& pRange) {
    for (std::size_t i = 0; i < cloud->size(); i ++) {
        Point temp(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        pRange.insert(temp);
    }
}

/** \brief Decide the resolution for supervoxel generation by random sampling*/
template<typename pointT>
void pc_resolution_sampling(typename PCT<pointT>::Ptr &cloud, std::vector<float> &reso, int sample_num) {
    float v_multi = 2.0f;
    float s_multi = 3.0f;
    
    pcl::RandomSample<pointT> rs;
    typename PCT<pointT>::Ptr sample(new PCT<pointT>);
    rs.setInputCloud(cloud);
    rs.setSample(sample_num);
    rs.setSeed(114514);
    rs.filter(*sample);

    pcl::KdTreeFLANN<pointT> nearest_search;
    nearest_search.setInputCloud(cloud);
    std::vector<int> pointIdxNearestSearch;
    std::vector<float> pointNearestSquaredDistance;
    int k = 10;
    
    float avg_dist = 0;
    for (std::size_t i = 0; i < sample_num; ++ i) {
        pointT searchPoint = sample->points[i];
        nearest_search.nearestKSearch(searchPoint, k, pointIdxNearestSearch, pointNearestSquaredDistance);
        
        float sum = 0.0f;
        for (std::size_t idx = 0; idx < k; ++ idx) {
            sum += sqrt(pointNearestSquaredDistance[idx]);
        }
        
        avg_dist += (sum / float(k)) / float(sample_num);
    }
    
    std::cout <<"Sampled reso.: " <<avg_dist <<std::endl;
    
    reso.resize(3);
    reso[0] = avg_dist * v_multi;
    reso[1] = avg_dist * s_multi;
    reso[2] = avg_dist;
}

/** \brief Generate supervoxels from a PCL point cloud*/
template<typename pointT>
SV_map<pointT> sv_generation(typename PCT<pointT>::Ptr &cloud, SV_clustering<pointT> &sv_clustering, float voxel_reso, float seed_reso) {
    float color_importance = 0.2f;
    float spatial_importance = 0.8f;
    float normal_importance = 1.0f;
    
    std::cout <<"Extracting SVs... \n" << std::endl;
    
//    pc_resolution_sampling<pointT>(cloud, reso, 200);
//    std::cout <<reso[0] <<" " <<reso[1] <<std::endl;
//    s.setVoxelResolution(reso[0]);
//    s.setSeedResolution(reso[1]);
    SV_clustering<pointT> s(voxel_reso, seed_reso);
    SV_map<pointT> smap;
    
    s.setInputCloud(cloud);
    s.setColorImportance(color_importance);
    s.setSpatialImportance(spatial_importance);
    s.setNormalImportance(normal_importance);
    s.extract(smap);
    
    std::cout << smap.size() <<" SVs Extracted. \n" << std::endl;
    sv_clustering = s;
    return smap;
    
}

/** \brief Generate supervoxels from a PCL point cloud
           return: minMax: {x_min, x_max, y_min, y_max}*/
template<typename pointT>
void cloud_xy_boundary(typename PCT<pointT>::Ptr &cloud,
                       std::vector<double> &minMax) {
    std::vector<std::vector<double>> xy_val(2, std::vector<double>());
    for (const auto &p : *cloud) {
        xy_val[0].push_back(p.x);
        xy_val[1].push_back(p.y);
    }
    for (auto &vec : xy_val)
        std::sort(vec.begin(), vec.end());
    
    minMax.push_back(xy_val[0][0]); minMax.push_back(xy_val[0][xy_val[0].size() - 1]);
    minMax.push_back(xy_val[1][0]); minMax.push_back(xy_val[1][xy_val[1].size() - 1]);
}

/** \brief Divide a PCL point cloud into grids.
        cloud: the original point cloud,
        grids: a 2D vector with all minimum height value of grids
        minMax: {x_min, x_max, y_min, y_max}. */
template<typename pointT>
void grid_division(typename PCT<pointT>::Ptr &cloud,
                   std::vector<std::vector<double>> &grids,
                   std::vector<double> &minMax,
                   int grid_size = 30) {
    cloud_xy_boundary<pointT>(cloud, minMax);
    
    int row_num = (minMax[3] - minMax[2]) / grid_size + 1,
        col_num = (minMax[1] - minMax[0]) / grid_size + 1;
    std::vector<std::vector<double>> grids_val(row_num,
                                                std::vector<double>(col_num, DBL_MAX));
    for (const auto &p : *cloud) {
        int r_num = (p.y - minMax[2]) / grid_size,
            c_num = (p.x - minMax[0]) / grid_size;
        
        if (p.z < grids_val[r_num][c_num])
            grids_val[r_num][c_num] = p.z;
    }
    
    //smoothe
    std::vector<std::vector<std::size_t>> outlier_indices;
    std::vector<double> indices_average_height;
    for (std::size_t r = 1; r < grids_val.size() - 1; ++ r) {
        for (std::size_t c = 1; c < grids_val[0].size() - 1; ++ c) {
            double average_height_diff = abs(((grids_val[r][c] - grids_val[r - 1][c]) +
                                             (grids_val[r][c] - grids_val[r][c - 1]) +
                                             (grids_val[r][c] - grids_val[r + 1][c]) +
                                             (grids_val[r][c] - grids_val[r][c + 1]))
                                             / 4.0),
                   average_height = (grids_val[r - 1][c] + grids_val[r][c - 1] +
                                     grids_val[r + 1][c] + grids_val[r][c + 1]) / 4.0;

            if (average_height_diff > static_cast<double>(grid_size) * 0.017) {
                outlier_indices.push_back(std::vector<std::size_t>({r, c}));
                indices_average_height.push_back(average_height);
            }
        }
    }

    for (std::size_t idx = 0; idx < outlier_indices.size(); ++ idx)
        grids_val[outlier_indices[idx][0]][outlier_indices[idx][1]] = indices_average_height[idx];
//
//    PCT<pclPointRGB>::Ptr grid_cloud(new PCT<pclPointRGB>);
//    for (std::size_t r = 0; r < grids_val.size(); ++ r) {
//        for (std::size_t c = 0; c < grids_val[0].size(); ++ c) {
//            if (grids_val[r][c] < 1e9) {
//                grid_cloud->push_back(pclPointRGB(minMax[0] + grid_size * c,
//                                                  minMax[2] + grid_size * r,
//                                                  grids_val[r][c]));
//            } else {
//                grid_cloud->push_back(pclPointRGB(minMax[0] + grid_size * c,
//                                                  minMax[2] + grid_size * r,
//                                                  -2300));
//            }
//        }
//    }
//
//    pcl::PLYWriter writer;
//    writer.write("/Users/konialive/Library/CloudStorage/GoogleDrive-lfliao0525@gmail.com/My Drive/fused_for_seg/SHRed/fixed_focal_length_winSize5_feature_bias/featureBias_grid_cloud.ply", *grid_cloud);
    
    grids = grids_val;
}

/** \brief Filter out all voxel units with less than 3 points (since unable to calculate normal, planarity, etc.).*/
template<typename pointT>
void minor_voxel_filtering(SV_map<pointT> &clusters,
                           std::multimap<std::uint32_t, std::uint32_t> &adjacency_map,
                           int max_label) {
    std::vector<std::uint32_t> outliers;
    SV_map<pointT> filtered_clusters;
    for (const auto &cls : clusters) {
        if (cls.second->voxels_->size() < 3) {
            outliers.push_back(cls.first);
        } else {
            filtered_clusters.insert(cls);
        }
    }
    
    std::multimap<std::uint32_t, std::uint32_t> filtered_adjacency_map;
    for (auto &pair : adjacency_map) {
        if (std::find(outliers.begin(), outliers.end(), pair.first) == outliers.end()
            and std::find(outliers.begin(), outliers.end(), pair.second) == outliers.end()
            and pair.first > 0 and pair.first <= max_label
            and pair.second > 0 and pair.second <= max_label)
            filtered_adjacency_map.insert(pair);
    }
    
    clusters = filtered_clusters;
    adjacency_map = filtered_adjacency_map;
    
    //validation
//    for (const auto &vid : outliers) {
//        std::cout <<vid <<"\n";
//    }
}

/** \brief Define the convexity of a connection.
           Return: 0 - convex; 1 - concave; 2 - plane. */
template<typename pointT>
int convexConcaveAdjacency(typename pcl::Supervoxel<pointT>::Ptr &central_voxel,
                            typename pcl::Supervoxel<pointT>::Ptr &adjacent_voxel) {
    pclPointRGB central_centroid = central_voxel->centroid_,
                adj_centroid = adjacent_voxel->centroid_;
    
    //vector from centre to adja.
    Eigen::Vector3d diff_vector({
        adj_centroid.x - central_centroid.x,
        adj_centroid.y - central_centroid.y,
        adj_centroid.z - central_centroid.z
    }),
    central_normal = pcl_normal_to_eigen_vector(central_voxel->normal_),
    adjacent_normal = pcl_normal_to_eigen_vector(adjacent_voxel->normal_);
    
    double central_angle = eigen_vector3d_angle(central_normal, diff_vector),
           adjacent_angle = eigen_vector3d_angle(adjacent_normal, diff_vector),
           diff_angle = eigen_vector3d_angle(central_normal, adjacent_normal);
    
    if (diff_angle > 10.0) {
        if (central_angle > adjacent_angle) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return -1;
    }
}

template<typename pointT>
double maximum_color_gradient(typename PCT<pointT>::Ptr &cloud) {
    pcl::KdTreeFLANN<pointT> nearest_tree;
    int k = 2;
    
    nearest_tree.setInputCloud(cloud);
    
    double maximum_gradient = -1;
    for (const auto &p : cloud->points) {
        std::vector<int> pointIdxKNNSearch(k);
        std::vector<float> pointKNNSquaredDistance(k);
        
        if (nearest_tree.nearestKSearch(p, k, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
            double gradient = (abs(p.r - (*cloud)[pointIdxKNNSearch[1]].r) +
                              abs(p.g - (*cloud)[pointIdxKNNSearch[1]].g) +
                              abs(p.b - (*cloud)[pointIdxKNNSearch[1]].b)) / 3;
            
            if (gradient >= maximum_gradient)
                maximum_gradient = gradient;
        }
    }
    
    return maximum_gradient;
}

// ===========================================================================
/** \brief Customed version of Alpha-expansion. */
namespace CGAL {

template <typename InputGraph,
          typename EdgeCostMap,
          typename VertexLabelCostMap,
          typename VertexLabelMap,
          typename NamedParameters = parameters::Default_named_parameters>
double my_alpha_expansion_graphcut (const InputGraph& input_graph,
                                 EdgeCostMap edge_cost_map,
                                 VertexLabelCostMap vertex_label_cost_map,
                                 VertexLabelMap vertex_label_map,
                                 const NamedParameters& np = parameters::default_values())
{
  using parameters::choose_parameter;
  using parameters::get_parameter;

  typedef boost::graph_traits<InputGraph> GT;
  typedef typename GT::edge_descriptor input_edge_descriptor;
  typedef typename GT::vertex_descriptor input_vertex_descriptor;

  typedef typename GetInitializedVertexIndexMap<InputGraph, NamedParameters>::type VertexIndexMap;
  VertexIndexMap vertex_index_map = CGAL::get_initialized_vertex_index_map(input_graph, np);

  typedef typename GetImplementationTag<NamedParameters>::type Impl_tag;

  // select implementation
  typedef typename std::conditional
    <std::is_same<Impl_tag, Alpha_expansion_boost_adjacency_list_tag>::value,
     Alpha_expansion_boost_adjacency_list_impl,
     typename std::conditional
     <std::is_same<Impl_tag, Alpha_expansion_boost_compressed_sparse_row_tag>::value,
      Alpha_expansion_boost_compressed_sparse_row_impl,
      Alpha_expansion_MaxFlow_impl>::type>::type
    Alpha_expansion;

  typedef typename Alpha_expansion::Vertex_descriptor Vertex_descriptor;

  Alpha_expansion alpha_expansion;

  // TODO: check this hardcoded parameter
  const double tolerance = 1e-10;

  double min_cut = (std::numeric_limits<double>::max)();

#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
  double vertex_creation_time, edge_creation_time, cut_time;
  vertex_creation_time = edge_creation_time = cut_time = 0.0;
#endif

  std::vector<Vertex_descriptor> inserted_vertices;
  inserted_vertices.resize(num_vertices (input_graph));

  std::size_t number_of_labels = get(vertex_label_cost_map, *(vertices(input_graph).first)).size();

  bool success;
  do {
    success = false;

    for (std::size_t alpha = 0; alpha < number_of_labels - 2; ++ alpha)
    {
      alpha_expansion.clear_graph();

#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
      Timer timer;
      timer.start();
#endif

      // For E-Data
      // add every input vertex as a vertex to the graph, put edges to source & sink vertices
      for (input_vertex_descriptor vd : CGAL::make_range(vertices(input_graph)))
      {
        std::size_t vertex_i = get(vertex_index_map, vd);
        Vertex_descriptor new_vertex = alpha_expansion.add_vertex();
        inserted_vertices[vertex_i] = new_vertex;
        double source_weight = get(vertex_label_cost_map, vd)[alpha];
        // since it is expansion move, current alpha labeled vertices will be assigned to alpha again,
        // making sink_weight 'infinity' guarantee this.
        double sink_weight = (std::size_t(get(vertex_label_map, vd)) == alpha ?
                              (std::numeric_limits<double>::max)()
                              : get(vertex_label_cost_map, vd)[get(vertex_label_map, vd)]);

        alpha_expansion.add_tweight(new_vertex, source_weight, sink_weight);
      }
#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
      vertex_creation_time += timer.time();
      timer.reset();
#endif

      // For E-Smooth
      // add edge between every vertex,
      for (input_edge_descriptor ed : CGAL::make_range(edges(input_graph)))
      {
        input_vertex_descriptor vd1 = source(ed, input_graph);
        input_vertex_descriptor vd2 = target(ed, input_graph);
        std::size_t idx1 = get (vertex_index_map, vd1);
        std::size_t idx2 = get (vertex_index_map, vd2);

        double weight = get (edge_cost_map, ed);

        Vertex_descriptor v1 = inserted_vertices[idx1],
          v2 = inserted_vertices[idx2];

        std::size_t label_1 = get (vertex_label_map, vd1);
        std::size_t label_2 = get (vertex_label_map, vd2);
        if(label_1 == label_2) {
          if(label_1 != alpha) {
            alpha_expansion.add_edge(v1, v2, weight, weight);
          }
        } else {
          Vertex_descriptor inbetween = alpha_expansion.add_vertex();

          double w1 = (label_1 == alpha) ? 0 : weight;
          double w2 = (label_2 == alpha) ? 0 : weight;
          alpha_expansion.add_edge(inbetween, v1, w1, w1);
          alpha_expansion.add_edge(inbetween, v2, w2, w2);
          alpha_expansion.add_tweight(inbetween, 0., weight);
        }
      }
#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
      edge_creation_time += timer.time();
#endif

      alpha_expansion.init_vertices();

#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
      timer.reset();
#endif

      double flow = alpha_expansion.max_flow();

#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
      cut_time += timer.time();
#endif

      if(min_cut - flow <= flow * tolerance) {
        continue;
      }
      min_cut = flow;
      success = true;
      //update labeling
      for (input_vertex_descriptor vd : CGAL::make_range(vertices (input_graph)))
      {
        std::size_t vertex_i = get (vertex_index_map, vd);
        alpha_expansion.update(vertex_label_map, inserted_vertices, vd, vertex_i, alpha);
      }
    }
  } while(success);

#ifdef CGAL_SEGMENTATION_BENCH_GRAPHCUT
  CGAL_TRACE_STREAM << "vertex creation time: " << vertex_creation_time <<
    std::endl;
  CGAL_TRACE_STREAM << "edge creation time: " << edge_creation_time << std::endl;
  CGAL_TRACE_STREAM << "max flow algorithm time: " << cut_time << std::endl;
#endif

  return min_cut;
}

}

// ===========================================================================

/** \brief Calculate the angle between PCL 3D vector and horizontal plane*/
template<typename normalT>
double vector_horizon_angle(normalT& n) {
    return (n.normal_x * 0.0 + n.normal_y * 0.0 + n.normal_z * 1.0) / sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y + n.normal_z * n.normal_z);
}

/** \brief Transform PCL normal vector to Eigen Vector3d. */
template<typename normalT>
Eigen::Vector3d pcl_normal_to_eigen_vector(normalT &n) {
    return Eigen::Vector3d({
        n.normal_x,
        n.normal_y,
        n.normal_z
    });
}

/** \brief Transform PCL normal vector to Eigen Vector3d. */
double eigen_vector3d_angle(Eigen::Vector3d &v1, Eigen::Vector3d &v2) {
    return std::atan2(v1.cross(v2).norm(), v1.dot(v2)) * 57.2957795;
}

/** \brief Find if a PCL point exists in the given vector*/
template<typename pointT>
const bool find_pcl_point(std::vector<pointT>& vec, pointT& p) {
    for (auto i : vec) {
        if (i.x == p.x && i.y == p.y && i.z == p.z)
            return true;
    }
    
    return false;
}

/** \brief Compare two PCL points. */
bool compare_pcl_point (pclPointRGB& p1, pclPointRGB& p2) {
    if (p1.x != p2.x)
        return p1.x > p2.x;
    else if (p1.y != p2.y)
        return p1.y > p2.y;
    else
        return p1.z > p2.z;
}

/** \brief Judge if two PCL points are equal. */
bool equal_pcl_point (pclPointRGB& p1, pclPointRGB& p2) {
    if (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z)
        return true;
    return false;
}

/** \brief Calculate the Euclidean distance between two vectors*/
double vector_euclidean_dist(Vector v1, Vector v2) {
    return sqrt(pow((v1[0] - v2[0]), 2.0) + pow((v1[1] - v2[1]), 2.0) + pow((v1[2] - v2[2]), 2.0));
}

/** \brief Calculate the Euclidean distance between two standard vectors. */
template<typename T>
double std_vector_euclidean_dist(std::vector<T> &v1, std::vector<T> &v2) {
    assert(v1.size() == v2.size());
    
    double accumulated_result = 0;
    for (std::size_t idx = 0; idx < v1.size(); ++ idx) {
//        accumulated_result += (v1[idx] - v2[idx]) *
//                                (v1[idx] - v2[idx]);
        
        //weighted
        if (idx == 0) {
            accumulated_result += (v1[idx] - v2[idx]) *
                                    (v1[idx] - v2[idx]) * pow(8, 2);
//        } else if (idx == 1) {
//            accumulated_result += (v1[idx] - v2[idx]) *
//                                    (v1[idx] - v2[idx]) * pow(2, 2);
        } else {
            accumulated_result += (v1[idx] - v2[idx]) *
                                    (v1[idx] - v2[idx]);
        }
        
//        accumulated_result += (v1[idx] - v2[idx]) * (v1[idx] - v2[idx]);
    }
    
    return sqrt(accumulated_result);
}

/** \brief Calculate the Euclidean distance between two PCL points*/
template<typename pointT>
double point_euclidean_dist(pointT &p1, pointT &p2) {
    return sqrt(pow((p1.x - p2.x), 2.0) + pow((p1.y - p2.y), 2.0) + pow((p1.z - p2.z), 2.0));
}

template<typename pointT>
double voxel_euclidean_dist(typename pcl::Supervoxel<pointT>::Ptr &sv1,
                            typename pcl::Supervoxel<pointT>::Ptr &sv2,
                            double voxel_reso) {
    double x1 = 0, y1 = 0, z1 = 0, x2 = 0, y2 = 0, z2 = 0;
    for (const auto &p : *sv1->voxels_) {
        x1 += p.x / sv1->voxels_->size();
        y1 += p.y / sv1->voxels_->size();
        z1 += p.z / sv1->voxels_->size();
    }
    for (const auto &p : *sv2->voxels_) {
        x2 += p.x / sv2->voxels_->size();
        y2 += p.y / sv2->voxels_->size();
        z2 += p.z / sv2->voxels_->size();
    }
    
    pclPointRGB p1(x1, y1, z1), p2(x2, y2, z2);
    return point_euclidean_dist(p1, p2) / voxel_reso;
}

template<typename pointT>
double voxel_color_dist(typename pcl::Supervoxel<pointT>::Ptr &sv1,
                            typename pcl::Supervoxel<pointT>::Ptr &sv2) {
    double r1 = 0, g1 = 0, b1 = 0, r2 = 0, g2 = 0, b2 = 0;
    for (const auto &p : *sv1->voxels_) {
        r1 += p.r / sv1->voxels_->size();
        g1 += p.g / sv1->voxels_->size();
        b1 += p.b / sv1->voxels_->size();
    }
    for (const auto &p : *sv2->voxels_) {
        r2 += p.r / sv2->voxels_->size();
        g2 += p.g / sv2->voxels_->size();
        b2 += p.b / sv2->voxels_->size();
    }
    
    return sqrt((r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2) / 3.0); //maximum_color_gradient *
}

template<typename pointT>
double voxel_normal_dist(typename pcl::Supervoxel<pointT>::Ptr &sv1,
                        typename pcl::Supervoxel<pointT>::Ptr &sv2) {
    Eigen::Vector3d n1 = pcl_normal_to_eigen_vector(sv1->normal_),
                    n2 = pcl_normal_to_eigen_vector(sv2->normal_);
    double normal_angle = eigen_vector3d_angle(n1, n2);
    
    return (normal_angle / 57.2957795) / M_PI;
}

/** \brief Identify outliers in a vector */
template <typename elementT>
std::vector<std::size_t> interquartile_outlier_detection (std::vector<elementT>& vec) {
    std::sort(vec.begin(), vec.end());
    std::vector<std::size_t> outlier_idx;
    double vector_1q = 0.0, vector_3q = 0.0, vector_iqr = 0.0;
    
    switch (vec.size()) {
        case 3:
            vector_1q = (vec[0] + vec[1]) / 2, vector_3q = (vec[1] + vec[2]) / 2;
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 4:
            vector_1q = vec[1], vector_3q = vec[2];
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 5:
            vector_1q = vec[1], vector_3q = vec[3];
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 6:
            vector_1q = (vec[1] + vec[2]) / 2, vector_3q = (vec[3] + vec[4]) / 2;
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 7:
            vector_1q = (vec[1] + vec[2]) / 2, vector_3q = (vec[4] + vec[5]) / 2;
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 8:
            vector_1q = vec[2], vector_3q = vec[5];
            vector_iqr = vector_3q - vector_1q;
            break;
            
        case 9:
            vector_1q = vec[2], vector_3q = vec[6];
            vector_iqr = vector_3q - vector_1q;
            break;
            
        default:
            break;
            
    }
    
//    std::cout <<vector_1q <<" " <<vector_3q <<" " <<vector_iqr <<std::endl;
    for (std::size_t idx = 0; idx < vec.size(); ++ idx) {
        if (vec[idx] <= vector_1q - 1.5 * vector_iqr || vec[idx] >= vector_3q + 1.5 * vector_iqr) {
            outlier_idx.push_back(idx);
        }
    }
    return outlier_idx;
}

/** \brief Remove lower outliers with SOR filters and elevation. */
template <typename pointT>
void lower_outliers_removal (typename PCT<pointT>::Ptr &cloud) {
    typename pcl::StatisticalOutlierRemoval<pointT>::Ptr sor_filter(new pcl::StatisticalOutlierRemoval<pointT>);
    sor_filter->setInputCloud(cloud);
    sor_filter->setMeanK(30);
    sor_filter->setStddevMulThresh(0.3);
    
    typename PCT<pointT>::Ptr filtered_cloud(new PCT<pointT>);
    typename PCT<pointT>::Ptr scatter_filtered_cloud(new PCT<pointT>);
    
    sor_filter->filter(*filtered_cloud);
    sor_filter->setNegative(true);
    sor_filter->filter(*scatter_filtered_cloud);
    
    std::vector<pointT> outliers;
    for (const auto &p : scatter_filtered_cloud->points)
        outliers.push_back(p);
    
    std::sort(outliers.begin(), outliers.end(),
              [](const pointT &a, pointT &b) -> bool {
        return a.z > b.z;
    });
    
    std::size_t total_outlier_num = outliers.size();
    while (outliers.size() > static_cast<int>(0.96 * total_outlier_num))
        outliers.pop_back();
    
    for (const auto &p : outliers) {
        filtered_cloud->push_back(p);
    }

    cloud = filtered_cloud;
}

/** \brief Apply the pyramid hierarchical constraint to adjacent level of grids */
template <typename pointT>
void pyramid_hierarchical_constraint (std::vector<std::vector<pointT>>& cloud_raster_dem, std::vector<std::vector<pointT>>& cloud_raster, std::size_t row, std::size_t col, double z_thres) {
    std::size_t prev_row = row / 2, prev_col = col / 2;
//    std::vector<double> lower_level_elevation;
    
    if (cloud_raster_dem[prev_row][prev_col].z != 10000) { //Examine if the dem is valid
        if (cloud_raster[row][col].z - cloud_raster_dem[prev_row][prev_col].z > 2 * z_thres) {
            cloud_raster[row][col].z = 10000;
        }
        return;
    } else {
        cloud_raster[row][col].z = 10000;
        return;
    }
}

/** \brief a three-dimension clustering of vectors in eight quadrants divided by Cartesian Coordinates*/
template<typename pointT>
void eight_quadrant_clustering(my_saliency<pointT>& s) {
    std::vector<std::vector<Vector>> v;
    Vector_map m = s.get_normal_map();
    for (int itr = 0; itr < s.getPts().size(); ++itr) {
        if (m[itr].x() >= 0) {
            if (m[itr].y() >= 0) {
                if (m[itr].z() >= 0) {
                    v[0].push_back(m[itr]);
                }else{
                    v[1].push_back(m[itr]);
                }
            }else{
                if (m[itr].z() >= 0) {
                    v[2].push_back(m[itr]);
                }else{
                    v[3].push_back(m[itr]);
                }
            }
        }else{
            if (m[itr].y() >= 0) {
                if (m[itr].z() >= 0) {
                    v[4].push_back(m[itr]);
                }else{
                    v[5].push_back(m[itr]);
                }
            }else{
                if (m[itr].z() >= 0) {
                    v[6].push_back(m[itr]);
                }else{
                    v[7].push_back(m[itr]);
                }
            }
        }
    }
    
    
    
}


#endif /* algo_in_class_h */
