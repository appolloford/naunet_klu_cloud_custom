set(PYMODNAME py${CMAKE_PROJECT_NAME} CACHE STRING "")

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

file(GLOB_RECURSE PYNAUNETFILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" *.py)
message(STATUS "PYNAUNETFILES                 = ${PYNAUNETFILES}")

foreach(name ${PYNAUNETFILES})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${name}" "${CMAKE_CURRENT_BINARY_DIR}/${name}")
endforeach()

add_subdirectory(pynaunet_model)

install(
    CODE 
    "execute_process(COMMAND \
    ${Python3_EXECUTABLE} -m pip install ${CMAKE_CURRENT_BINARY_DIR} \
    --prefix ${CMAKE_INSTALL_PREFIX})"
)
