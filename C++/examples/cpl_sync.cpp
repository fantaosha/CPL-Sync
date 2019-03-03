#include <CPLSync/CPLSync.h>
#include <SESync/SESync.h>

#include <fstream>

int main(int argc, char *argv[]) {
  bool use_CPL = true;
  bool write = false;

  if (argc < 2 || argc > 4) {
    std::cout << "Usage: " << argv[0] << " [input .g2o file]" << std::endl;
    exit(1);
  }

  if (argc >= 3) {
    std::string alg(argv[2]);

    if (alg == "CPL") {
      use_CPL = true;
    } else if (alg == "SE") {
      use_CPL = false;
    } else {
      std::cout << "The second argument must be either \"CPL\" or \"SE\""
                << std::endl;
      exit(1);
    }
  }

  if (argc >= 4) {
    write = true;
  }

  size_t num_poses;

  if (use_CPL) {
    // PGO with CPL-Sync
    CPLSync::measurements_t measurements =
        CPLSync::read_g2o_file(argv[1], num_poses);

    CPLSync::CPLSyncOpts opts;
    opts.verbose = true; // Print output to stdout
    opts.reg_Cholesky_precon_max_condition_number = 2e6;

#if defined(_OPENMP)
    opts.num_threads = 4;
#endif

    CPLSync::CPLSyncResult results = CPLSync::CPLSync(measurements, opts);

    if (write) {
      std::string filename(argv[3]);
      std::cout << "Saving final poses to file: " << filename << std::endl;
      std::ofstream poses_file(filename);
      poses_file << results.xhat;
      poses_file.close();
    }
  } else {
    // PGO with SE-Sync
    SESync::measurements_t measurements =
        SESync::read_g2o_file(argv[1], num_poses);

    SESync::SESyncOpts opts;
    opts.verbose = true; // Print output to stdout
    opts.reg_Cholesky_precon_max_condition_number = 2e6;

#if defined(_OPENMP)
    opts.num_threads = 4;
#endif

    SESync::SESyncResult results = SESync::SESync(measurements, opts);

    if (write) {
      std::string filename(argv[3]);
      std::cout << "Saving final poses to file: " << filename << std::endl;
      std::ofstream poses_file(filename);
      poses_file << results.xhat;
      poses_file.close();
    }
  }

  return 0;
}
