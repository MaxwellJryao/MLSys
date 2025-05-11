aux_source_directory(. LIB_LIST)

add_library(MathFunctions STATIC ${LIB_LIST})

# 设置库文件的输出目录

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)