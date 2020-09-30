add_executable(Fractals
  ComplexRange.swift
  ImageUtilities.swift
  JuliaSet.swift
  JuliaSetP.swift
  MandelbrotSet.swift
  main.swift)
target_link_libraries(Fractals PRIVATE
  ArgumentParser
  ModelSupport
  SwiftRT)


install(TARGETS Fractals
  DESTINATION bin)
