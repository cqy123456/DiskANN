#pragma once
#include <sys/resource.h>
#include <unistd.h>

#include <iostream>

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in MB, or zero if the value cannot be
 * determined on this OS.
 */
inline uint64_t getPeakRSS() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return static_cast<uint64_t>(rusage.ru_maxrss * 1024LL / 1000000);
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in MB, or zero if the value cannot be determined on this OS.
 */
inline uint64_t getCurrentRSS() {
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return static_cast<uint64_t>(0LL); /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return static_cast<uint64_t>(0LL); /* Can't read? */
  }
  fclose(fp);
  return static_cast<uint64_t>(rss * sysconf(_SC_PAGESIZE) / 1000000);
}