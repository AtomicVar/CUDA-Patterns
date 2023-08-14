//
//  Timer.h
//  NLE
//
//  Created by Guo Shuai on 2023/7/27.
//

#ifndef Timer_h
#define Timer_h

#include <chrono>
#include <string>
#include <iostream>
#include <unordered_map>

class Timer {
  
  struct Stat {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    long long total_duration = 0;
    long long mean_duration = 0;
    int count = 0;
  };
  
public:
  // Get the Timer Singleton
  static Timer* Get() {
    static Timer timer;
    return &timer;
  }
  
  // Print all mean durations
  void PrintMeanTimes() {
    for (const auto& [name, stat] : stats_) {
      printf("[Timer Summary] %s mean time: %lldus\n", name.c_str(), stat.total_duration / stat.count);
    }
  }
  
  void Start(const std::string& name) {
    stats_[name].start = std::chrono::high_resolution_clock::now();
  }
  
  void End(const std::string& name) {
    auto end = std::chrono::high_resolution_clock::now();
    auto start = stats_[name].start;
    long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    stats_[name].total_duration += duration;
    stats_[name].count++;
    printf("[Timer] %s time: %lldus\n", name.c_str(), duration);
  }
  
private:
  std::unordered_map<std::string, Stat> stats_;
};

#endif /* Timer_h */