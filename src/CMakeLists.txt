set(
    OBJTARGETS
        naunet_constants
        naunet_physics
        naunet_renorm
        naunet_rates
        naunet_fex
        naunet_jac
)

# check enabled languages
get_property(
    languages
        GLOBAL
            PROPERTY ENABLED_LANGUAGES
)

# Functions from Numerical Recipes, not suppport GPU yet.
if(NOT "CUDA" IN_LIST languages)
    list(
        APPEND OBJTARGETS
            naunet_utilities
    )
endif()

foreach(name ${OBJTARGETS})
    add_library(
        ${name}
            OBJECT
            ${name}.cpp
    )
    target_include_directories(
        ${name}
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(
        ${name}
            ${SUNDIALSLIB}
    )
endforeach()

if(MAKE_SHARED)
    add_library(
        naunet_shared
            SHARED
            naunet.cpp
    )
    set_target_properties(
        naunet_shared
            PROPERTIES
                OUTPUT_NAME naunet
    )
    target_include_directories(
        naunet_shared
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(
        naunet_shared
            ${OBJTARGETS}
            ${SUNDIALSLIB}
    )
    install(
        TARGETS naunet_shared
        LIBRARY
        DESTINATION lib
    )
endif()

if(MAKE_STATIC)
    add_library(
        naunet_static
            STATIC
            naunet.cpp
    )
    set_target_properties(
        naunet_static
            PROPERTIES
                OUTPUT_NAME naunet
    )
    target_include_directories(
        naunet_static
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(
        naunet_static
            ${OBJTARGETS}
            ${SUNDIALSLIB}
    )
    install(
        TARGETS naunet_static
        LIBRARY
        DESTINATION lib
    )
endif()


install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
)