//
// Created by 吴松林 on 2022/8/1.
//
#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false
#define INF 0xfffffff
namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

template<typename T>
int search_disk_segment(
    diskann::Metric& metric, const std::string& index_path_prefix, T* query,
    size_t query_num, size_t query_aligned_dim, const unsigned num_threads,
    const unsigned recall_at, const unsigned beamwidth,
    const unsigned num_nodes_to_cache, const std::vector<unsigned> Lvec,
    std::vector<std::vector<unsigned>> query_ids,
    std::vector<std::vector<float>>    query_dis,
    std::vector<diskann::QueryStats*> stats_Lvec, std::vector<double> diff) {
  std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader, metric));
  int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

  if (res != 0) {
    return res;
  }
  // cache bfs levels
  //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  if (num_nodes_to_cache > 0) {
    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache
                  << " BFS nodes around medoid(s)" << std::endl;
    _pFlashIndex->generate_cache_list_from_sample_queries(
        warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();
  }

  omp_set_num_threads(num_threads);

  uint32_t optimized_beamwidth = 2;
  optimized_beamwidth = beamwidth;

  for (unsigned test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    query_ids[test_id].clear();
    query_dis[test_id].clear();
    query_ids[test_id].resize(recall_at * query_num);
    query_dis[test_id].resize(recall_at * query_num);
    std::vector<uint64_t> ids_64(recall_at * query_num);
    auto&                 ids = query_ids[test_id];
    auto&                 dis = query_dis[test_id];
    auto                  stats = stats_Lvec[test_id];
    auto                  s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * query_aligned_dim), recall_at, L,
          ids_64.data() + (i * recall_at), dis.data() + (i * recall_at),
          optimized_beamwidth, false, stats + i);
    }
    auto e = std::chrono::high_resolution_clock::now();
    diff[test_id] += (e - s).count();
    diskann::convert_types<uint64_t, uint32_t>(ids_64.data(), ids.data(),
                                               query_num, recall_at);
  }
}

void merge_res(std::vector<unsigned>& ids_res, std::vector<float>& dis_res,
               std::vector<unsigned>& ids_query, std::vector<float>& dis_query,
               unsigned recall_at, unsigned query_num, unsigned offset) {
  std::vector<unsigned> ids_tmp(ids_res.size());
  std::vector<float>    dis_tmp(ids_res.size());
  for (unsigned i = 0; i < query_num; i++) {
    unsigned s = 0 + i * recall_at, t = 0 + i * recall_at;
    unsigned k = i * recall_at;
    for (unsigned j = 0; j < recall_at; j++) {
      if (dis_res[s] < dis_query[t]) {
        ids_tmp[j + k] = ids_res[s];
        dis_tmp[j + k] = dis_res[s];
        s++;
      } else {
        ids_tmp[j + k] = ids_query[t] + offset;
        dis_tmp[j + k] = dis_query[t];
        t++;
      }
    }
  }
  ids_res.swap(ids_tmp);
  dis_res.swap(dis_tmp);
}

template<typename T>
int search_disk_index(
    diskann::Metric& metric, const std::vector<std::string>& disk_segments,
    const std::string& result_output_prefix, const std::string& query_file,
    std::string& gt_file, const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const std::vector<unsigned>& Lvec, const bool use_reorder_data = false) {
  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::endl;

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }
  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
                << std::setw(16) << "99.9 Latency" << std::setw(16)
                << "Mean IOs" << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(16) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "======================================================="
      << std::endl;
  std::vector<std::vector<unsigned>> query_ids(Lvec.size());
  std::vector<std::vector<float>>    query_dis(Lvec.size());

  std::vector<std::vector<unsigned>> res_ids(Lvec.size());
  std::vector<std::vector<float>>    res_dis(Lvec.size());

  std::vector<diskann::QueryStats*> stats_Lvec(Lvec.size());
  std::vector<double>               diff_Lvec(Lvec.size());
  for (auto& stats : stats_Lvec) {
    stats = new diskann::QueryStats[query_num];
  }
  unsigned offset = 0;
  for (unsigned i = 0; i < disk_segments.size(); i++) {
    diskann::cout << "searching on segment " << i << std::endl;
    auto disk_segment = disk_segments[i];
    search_disk_segment(metric, disk_segment, query, query_num,
                        query_aligned_dim, num_threads, recall_at, beamwidth,
                        num_nodes_to_cache, Lvec, query_ids, query_dis,
                        stats_Lvec, diff_Lvec);
    diskann::cout << "search done on segment " << i << std::endl;
    if (i == 0) {
      for (unsigned j = 0; j < Lvec.size(); j++) {
        res_ids.resize(recall_at*query_num);
        res_dis.resize(recall_at*query_num);
        res_ids[j].swap(query_ids[j]);
        res_dis[j].swap(query_dis[j]);
      }
    } else {
      for (unsigned j = 0; j < Lvec.size(); j++) {
        merge_res(res_ids[j], res_dis[j], query_ids[j], query_dis[j], recall_at,
                  query_num, offset);
      }
    }
    uint64_t      npts;
    std::ifstream reader(disk_segment);
    reader.read((char*) &npts, sizeof(uint64_t));
    reader.close();
    offset += npts;
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    auto& diff = diff_Lvec[test_id];
    float qps = (1.0 * query_num) / (1.0 * diff);
    auto& stats = stats_Lvec[test_id];

    auto mean_latency = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    auto latency_999 = diskann::get_percentile_stats<float>(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    auto mean_cpuus = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });

    float recall = 0;
    if (calc_recall_flag) {
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         res_ids[test_id].data(), recall_at,
                                         recall_at);
    }
    auto L = Lvec[test_id];

    diskann::cout << std::setw(6) << L << std::setw(12) << beamwidth
                  << std::setw(16) << qps << std::setw(16) << mean_latency
                  << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                  << std::setw(16) << mean_cpuus;
    if (calc_recall_flag) {
      diskann::cout << std::setw(16) << recall << std::endl;
    } else
      diskann::cout << std::endl;
    delete[] stats;
  }
  diskann::aligned_free(query);
}

int main(int argc, char** argv) {
  std::string              data_type, dist_fn, query_file, gt_file;
  std::string              result_path_prefix;
  unsigned                 num_threads, K, W, num_nodes_to_cache;
  std::vector<unsigned>    Lvec;
  std::vector<std::string> segment_files;
  bool                     use_reorder_data = false;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()(
        "segments_path_prefix",
        po::value<std::vector<std::string>>(&segment_files)->multitoken(),
        "Path prefix to the segments");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->required(),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Beamwidth for search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);

  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float"))
      return search_disk_index<float>(
          metric, segment_files, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    else if (data_type == std::string("int8"))
      return search_disk_index<int8_t>(
          metric, segment_files, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    else if (data_type == std::string("uint8"))
      return search_disk_index<uint8_t>(
          metric, segment_files, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}