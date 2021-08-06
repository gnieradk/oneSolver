/*! Example of MPI simple program with SYCL integration
 *
 * This is an example program of using MPI with the integration
 * of SYCL. This example program is borrowed from Intel web page:
 * https://software.intel.com/content/www/us/en/develop/articles/compile-and-run-mpi-programs-using-dpcpp-language.html
 *
 * Even a better place of referenc of this program is given here:
 * https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC++/ParallelPatterns/dpc_reduce
 *
 */
 
 #include <oneapi/dpl/execution>
 #include <oneapi/dpl/numeric>
 
 #include <mpi.h>
 #include <CL/sycl.hpp>
 
 #include <iostream>
 #include <iomanip>

 
 #include "dpc_common.hpp"
 
 using namespace sycl;
 constexpr int master = 0;
 
 
 void mpi_native(float* results, int rank_num, int num_procs,
                 long total_num_steps, queue& q){
    float dx, dx2;
    dx = 1.0f / (float)total_num_steps;
    dx2 = dx / 2.0f;
    
    default_selector device_selector;
    
    try {
      // The size of amount of memory that will be given to the buffer.
      range<1> num_items{total_num_steps / size_t(num_procs)};
      
      // Buffers are used to tell SYCL which data will be shared between the host
      // and the devices.
      
      buffer<float, 1> results_buf(results,
                                  range<1>(total_num_steps / size_t(num_procs)));
                                  
      // Submit takes in a lambda that is passed in a command group handler
      // constructed at runtime.
      q.submit([&](handler& h){
          // Accessors are used to get access to the meory owned by the buffers.
          accessor results_accessor(results_buf, h, write_only);
          // Each kernel calculates a partial of the number Pi in parallel.
          h.parallel_for(num_items, [=](id<1> k){
              float x = ((float)rank_num / (float)num_procs) + (float)k * dx + dx2;
              results_accessor[k] = (4.0f * dx) / (1.0f + x * x);
              });
          });
      } catch (...) {
          std::cout << "Failure" << std::endl;
      }
                     
 }
 
 
 
 int main(int argc, char** argv){
     int num_steps = 1000000;
     int groups = 10000;
     char machine_name[MPI_MAX_PROCESSOR_NAME];
     int name_len = 0;
     int id = 0;
     int num_procs = 0;
     float pi=0.0;
     queue myQueue{property::queue::in_order()};
     auto policy = oneapi::dpl::execution::make_device_policy(
            queue(default_selector{}, dpc_common::exception_handler));
            
     // Start MPI
     if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
         std::cout << "Failed to initialize MPI\n";
         exit(-1);
     }
     
     // Create the communicator, and retrieve the number of MPI ranks.
     MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
     
     // Determine the rank number.
     MPI_Comm_rank(MPI_COMM_WORLD, &id);
     
     // Get the machine name.
     MPI_Get_processor_name(machine_name, &name_len);
     
     std::cout << "Rank #" << id << "runs on: " << machine_name
               << " uses device: "
               << myQueue.get_device().get_info<info::device::name>() << "\n";
               
     int num_step_per_rank = num_steps / num_procs;
     float* results_per_rank = new float[num_step_per_rank];
     
     // Initialize an array to store a partial result per rank.
     for (size_t i = 0; i < num_step_per_rank; i++) results_per_rank[i] = 0.0;
     
     dpc_common::TimeInterval TimeMPI;
     // Calculate the Pi number partially by multiple MPI ranks.
     mpi_native(results_per_rank, id, num_procs, num_steps, myQueue);
     
     float local_sum = 0.0;
     
     //Use the DPC++ library call to reduce the array using plus
     buffer<float> calc_values(results_per_rank, num_step_per_rank);
     auto calc_begin = oneapi::dpl::begin(calc_values);
     auto calc_end   = oneapi::dpl::end(calc_values);
     
     local_sum = 
        std::reduce(policy, calc_begin, calc_end, 0.0f, std::plus<float>());
        
     // Master rank performs a reduce operation to get the sum of all partial Pi.
     MPI_Reduce(&local_sum, &pi, 1, MPI_FLOAT, MPI_SUM, master, MPI_COMM_WORLD);
     
     if (id == master){
         auto stop = TimeMPI.Elapsed();
         std::cout << "mpi native:\t\t";
         std::cout << std::setprecision(6) << "PI =" << pi;
         std::cout << " in " << stop << "seconds\n";
     }
     
     delete[] results_per_rank;
     
     MPI_Finalize();
     
     return 0;
     
     
 }