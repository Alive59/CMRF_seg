//
//  DFS_search.h
//  CMRF_seg
//
//  Created by Alive on 2023/8/18.
//

#ifndef DFS_search_h
#define DFS_search_h

#include "algo_in_class.h"

class DFSVisitor : public boost::default_dfs_visitor {
public:
    void vertex_search(vertex_descriptor vd, const Graph &g) const {
        std::cout <<"Found " <<g[vd].vid <<"\n";
    }
};

#endif /* DFS_search_h */
