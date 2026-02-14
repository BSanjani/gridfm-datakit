import gridfm_datakit as gfd

# Load IEEE 14-bus case
case = gfd.network.load_net_from_pglib('case14_ieee')

print("=" * 60)
print("IEEE 14-BUS SYSTEM DATA")
print("=" * 60)

# Bus information
print(f"\nTotal buses: {len(case.buses)}")
print(f"Bus numbers: {case.buses[:, gfd.network.BUS_I].astype(int).tolist()}")

# Generator information
print(f"\nTotal generators: {len(case.gens)}")
print(f"Generator buses: {case.gens[:, gfd.network.GEN_BUS].astype(int).tolist()}")

# Detailed generator info
print("\nDetailed Generator Information:")
print("-" * 60)
for i, gen in enumerate(case.gens):
    bus_num = int(gen[gfd.network.GEN_BUS])
    pg = gen[gfd.network.PG]
    qg = gen[gfd.network.QG]
    print(f"Generator {i+1}: Bus {bus_num}, Pg={pg:.2f} MW, Qg={qg:.2f} MVAr")

# Bus types
print("\nBus Type Summary:")
print("-" * 60)
ref_buses = case.buses[case.buses[:, gfd.network.BUS_TYPE] == gfd.network.REF]
pv_buses = case.buses[case.buses[:, gfd.network.BUS_TYPE] == gfd.network.PV]
pq_buses = case.buses[case.buses[:, gfd.network.BUS_TYPE] == gfd.network.PQ]

print(f"Slack buses (REF): {len(ref_buses)}")
if len(ref_buses) > 0:
    print(f"  Bus numbers: {ref_buses[:, gfd.network.BUS_I].astype(int).tolist()}")

print(f"PV buses (generators): {len(pv_buses)}")
if len(pv_buses) > 0:
    print(f"  Bus numbers: {pv_buses[:, gfd.network.BUS_I].astype(int).tolist()}")

print(f"PQ buses (loads): {len(pq_buses)}")
if len(pq_buses) > 0:
    print(f"  Bus numbers: {pq_buses[:, gfd.network.BUS_I].astype(int).tolist()}")

print("\n" + "=" * 60)