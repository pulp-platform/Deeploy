# Library Structure

This repository contains the following folders:

```
deeploy
├── cmake
├── Deeploy
├── TargetLibraries
├── DeeployTest
├── docs
├── install
├── scripts
└── toolchain
```

The core abstractions and framework of Deeploy is contained in `Deeploy`. The folder `TargetLibraries` contains C microkernels for these platforms. `DeeployTest` contains the testing framework for Deeploy. The folders `install` and `toolchain` are used for local installations of the required compilation toolchain and its dependencies. `scripts` contains some helper scripts, mostly for code formatting. The `cmake` folder contains CMake configuration files which are used by the testing infrastructure to configure compiler flags and simulator targets.

## Deeploy

The Deeploy folder mainly contains the `DeeployTypes.py` and `AbstractDataTypes.py` files, which, in turn, contain the core abstractions of Deeploy. The remainder of the folder structure contains the `Target` folder and several extensions to `Deeploy`'s core flow, and appears as follows:

```
deeploy
├── Deeploy
	├── DeeployTypes.py
	├── AbstractDataTypes.py
	├── CommonExtensions
	├── EngineExtension
	├── FutureExtension
	├── MemoryLevelExtension
	├── Targets
	└── TilingExtension
```

### Targets

The `Targets` folder contains the Deeploy models and code generation infrastructure for a specific platform; currently, Deeploy supports the following targets:

```
deeploy
├── Deeploy
	├── Targets
		├── CortexM
		├── Generic
		├── MemPool
		├── Neureka
		└── PULPOpen
```

Each of these `Target` folders is internally structured as follows:

```
deeploy
├── Deeploy
	├── Targets
		├── PULPOpen
			├── Bindings.py
			├── DataTypes.py
			├── Deployer.py
			├── Layers.py
			├── Parsers.py
			├── Platform.py
			├── TypeCheckers.py
			├── Tiler.py
			├── TileConstraints
			├── CodeTransformationPasses
			├── TopologyOptimizationPasses
			└── Templates
```

Where, by convention, files ending with `.py` are implementations of either classes in `DeeployTypes.py`, `AbstractDataTypes.py`, or one of the extensions. For new platform contributions, please follow this general folder structure.

### Extensions

Each folder named `-Extension` contains widely reusable abstractions; they are internally structured like Targets, using names like `Bindings.py`, `DataTypes.py`, `Deployer.py`, `Layers.py`, `Parsers.py`, `Platform.py` and `TypeCheckers.py` for extensions concerning the appropriate base Deeploy abstraction. They may further add new filenames according to the need of the extension. For example, the `MemoryLevelExtension` is structured like this:

```
deeploy
├── Deeploy
	├── MemoryLevelExtension
		├── MemoryLevels.py
		├── NetworkDeployers
		└── OptimizationPasses
```

When adding new extensions, please try to structure them similiarly to the structure used for `Targets` and existing `Extension`s.
