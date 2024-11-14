execute_process(
  COMMAND hostname
  OUTPUT_VARIABLE HOSTNAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("Compilation sur: ${HOSTNAME}")

if(HOSTNAME STREQUAL "login1.int.inf5171.calculquebec.cloud")
  message(FATAL_ERROR "Ne pas compiler sur le noeud frontal. Utiliser: srun -c 8 ninja")
endif()
