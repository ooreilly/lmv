add_executable(test_dict test_dict.cpp)    
target_include_directories(test_dict PUBLIC ${LMV_INCLUDE_DIR})
add_test(test_dict test_dict)

add_executable(test_hArray test_hArray.cpp)    
target_include_directories(test_hArray PUBLIC ${LMV_INCLUDE_DIR})
add_test(test_hArray test_hArray)

add_executable(test_dArray test_dArray.cu)    
target_include_directories(test_dArray PUBLIC ${LMV_INCLUDE_DIR})
target_link_libraries(test_dArray cublas cusparse)
add_test(test_dArray test_dArray)

add_executable(test_hcsr test_hcsr.cpp)    
target_include_directories(test_hcsr PUBLIC ${LMV_INCLUDE_DIR})
add_test(test_hcsr test_hcsr)

add_executable(test_dcsr test_dcsr.cu)    
target_include_directories(test_dcsr PUBLIC ${LMV_INCLUDE_DIR})
target_link_libraries(test_dcsr cublas cusparse)
add_test(test_dcsr test_dcsr)

