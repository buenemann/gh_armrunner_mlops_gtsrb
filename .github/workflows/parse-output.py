import sys
openblas = sys.argv[1]
acl = sys.argv[2]

results_sep = "---------------------------------------------------------"
openblas = openblas.split(results_sep, 1)[-1]
openblas_cpu_time_total = openblas.split("Self CPU time total: ", 1)[-1]
acl = acl.split(results_sep, 1)[-1]
acl_cpu_time_total = acl.split("Self CPU time total: ", 1)[-1]

print("                         ------------------        OpenBLAS results       ------------------")
print(openblas)
print("\n")
print("                         ------------------  Arm Compute Library results  ------------------")
print(acl)
print("\n")
print("                         ------------------  Self CPU time total change   ------------------")
print(f"Self CPU time total went from {openblas_cpu_time_total} (OpenBLAS) to {acl_cpu_time_total} (ACL): {(float(acl_cpu_time_total.split("ms", 1)[0])/float(openblas_cpu_time_total.split("ms", 1)[0]))*100:.2f}% change")