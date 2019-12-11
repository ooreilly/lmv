#add_executable(read_config test_read_config.cu)    
#target_include_directories(read_config PUBLIC ${DCSRMV_SOURCE_DIR}/src/)
#target_link_libraries(read_config cublas cusparse)
#
#add_executable(test_array test_array.cu)    
#target_include_directories(test_array PUBLIC ${DCSRMV_SOURCE_DIR}/src/)
#target_link_libraries(test_array cublas cusparse)
#
#add_executable(test_csr test_csr.cu)    
#target_include_directories(test_csr PUBLIC ${DCSRMV_SOURCE_DIR}/src/)
#target_link_libraries(test_csr cublas cusparse)

#add_executable(test_config test_config.cpp)    
#target_include_directories(test_config PUBLIC ${DCSRMV_SOURCE_DIR}/src/)
#target_link_libraries(test_csr cublas cusparse)

add_executable(test_dict test_dict.cpp)    
target_include_directories(test_dict PUBLIC ${LMV_SOURCE_DIR}/src/)
add_test(test_dict test_dict)

add_executable(test_hArray test_hArray.cpp)    
target_include_directories(test_hArray PUBLIC ${LMV_SOURCE_DIR}/src/)
add_test(test_hArray test_hArray)

add_executable(test_dArray test_dArray.cu)    
target_include_directories(test_dArray PUBLIC ${LMV_SOURCE_DIR}/src/)
target_link_libraries(test_dArray cublas cusparse)
add_test(test_dArray test_dArray)

add_executable(test_hcsr test_hcsr.cpp)    
target_include_directories(test_hcsr PUBLIC ${LMV_SOURCE_DIR}/src/)
add_test(test_hcsr test_hcsr)

add_executable(test_dcsr test_dcsr.cu)    
target_include_directories(test_dcsr PUBLIC ${LMV_SOURCE_DIR}/src/)
target_link_libraries(test_dcsr cublas cusparse)
add_test(test_dcsr test_dcsr)

