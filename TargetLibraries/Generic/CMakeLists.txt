file(GLOB_RECURSE SOURCES 
    "src/**"
)

add_deeploy_library(deeploybasic STATIC ${SOURCES})

target_include_directories(deeploybasic 
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/inc
)
