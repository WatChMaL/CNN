from modulefinder import ModuleFinder

finder = ModuleFinder()
finder.run_script('../watchmal_cl.py')

print("Loaded {} modules".format(len(finder.modules.items())))

print('Loaded modules:')
for name, mod in finder.modules.items():
        print('%s: ' % name, end='')
        print(','.join(list(mod.globalnames.keys())[:3]))
        if mod.__file__ is not None:
             print("       {}".format(mod.__file__))
