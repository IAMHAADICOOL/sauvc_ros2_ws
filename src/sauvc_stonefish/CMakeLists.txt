cmake_minimum_required(VERSION 3.10)
project(sauvc_stonefish)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  scenarios
  data
  scripts
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/ardusub_json_bridge.py
  scripts/ardusub_move_example.py
  scripts/randomize_arena.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
