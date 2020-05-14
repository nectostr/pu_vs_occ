import DATA.create_synthetic as cs
import comparative_synth as comp

# # Pos size
# for size in (20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50):
#     cs.create(error=e)
#     comp.comare()

# # Error experiment
# for e in (0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5):
#     cs.create(error=e)
#     comp.comare()

# Close select experiment
cs.create()
comp.comare()
cs.create_close()
comp.comare()
cs.create_close(reverse=True)
comp.comare()