#### objects.py
# This file examines the core dunder methods: __new__, __init__, __str__,
# __repr__, and __slots__, and their practical use cases in Python's object model,
# especially with examples from the AI/ML domain.
#--------------------------------------------------------------------------

### A-) __new__ ve __init__: Object Creation Process
# A.1.1
## When an object is created with `Class()`, a two-stage process runs in Python.
## The two main actors of this process are the __new__ and __init__ methods.

# 1. Division of Labor: Creator and Initializer
# ----------------------------------------
# __new__: This is the "Creator". Its task is to allocate memory for an instance
#          of the desired class and create and return this empty object.
#          This method is the first step that ensures an object exists.
#
# __init__: This is the "Initializer" (also known as the constructor). Its task
#           is to take the empty object created by __new__ and set its initial
#           state with attributes. In other words, it "fills" the object with
#           values like `self.attribute = value`.

# 2. Mechanical and Technical Differences
# ----------------------------
# - Arguments: __new__ takes the class itself (`cls`) as its first argument.
#              __init__ takes the reference to the created object (`self`) as its first argument.
# - Return Value: __new__ MUST return an object reference.
#                 __init__ should not return anything (it returns `None`).
#                 Its task is only to modify the object given to it.
# - Call Type: __new__ is a class method (behaves like @classmethod),
#             __init__ is an instance method.

print("--- A.1.1) Basic Creation Process Demonstration ---")
class MyClass:
    def __new__(cls, *args, **kwargs):
        print("1. __new__ called: Object is being created in memory.")
        instance = super().__new__(cls)
        return instance

    def __init__(self):
        print("2. __init__ called: Created object is being initialized.")

my_obj = MyClass()
# Output:
# 1. __new__ called: Object is being created in memory.
# 2. __init__ called: Object is being initialized.

#--------------------------------------------------------------------------
### A.1.2 ve A.1.3) Practical Use Cases
## Generally, __init__ fulfills all our needs. However, in some special cases,
## we need to intervene in the object creation process with __new__.

# Scenario 1: Singleton Design Pattern
# Used when we want to guarantee that only a single instance of a class exists
# throughout the application.
# (E.g., loading a large ML model into memory only once, a shared database
# connection, a global configuration object, or a single copy of a large ML model
# in memory).
# With __new__, instead of creating a new object every time, we return the existing single instance.
print("\n--- A.1.2) Scenario: Singleton Design Pattern ---")

class GlobalModel:
    _instance = None
    def __new__(cls):
        if not cls._instance:
            print("Model is being loaded into memory for the first time...")
            # Assume the model is loaded from disk here
            cls._instance = super().__new__(cls)
        else:
            print("Existing model reference is being used...")
        return cls._instance

# Test the Singleton
model1 = GlobalModel()
model2 = GlobalModel()
print(f"Do both variables hold the same model object? -> {model1 is model2}")


# Scenario 2: Creating Immutable Classes
# Used when inheriting from immutable base types like `tuple` or `str`,
# whose content cannot be changed after creation. In such objects,
# since values cannot be changed once __init__ runs, they must be set
# during the __new__ stage (E.g., creating a hyperparameter set that
# needs to be immutable).
print("\n--- A.1.3) Scenario: Immutable Hyperparameter Set ---")

class HyperparameterSet(tuple):
    def __new__(cls, lr, epochs, optimizer):
        # Values are passed directly to the __new__ method of the parent
        # class (tuple) at the moment of creation.
        return super().__new__(cls, (lr, epochs, optimizer))

    def __init__(self, lr, epochs, optimizer):
        # The __init__ method is empty here because the object (tuple)
        # cannot be changed after creation. Values are already set in __new__.
        pass

params = HyperparameterSet(0.01, 100, 'adam')
print(f"Created parameter set: {params}")

try:
    params[0] = 0.005 # Let's try to change the learning rate
except TypeError as e:
    print(f"Assignment Error: '{type(params).__name__}' object does not support item assignment.")

#--------------------------------------------------------------------------
### B) __str__ ve __repr__: Object Representations
# In Python, our objects have two different "representation" forms. These methods
# determine how an object will appear as text in different situations.

# __str__: Produces a readable output for the "end-user". Its purpose is to provide
#           a nice and understandable representation. The `print()` function or
#           `str()` call directly uses this method. If __str__ is not defined,
#           Python automatically uses the __repr__ method.

# __repr__: Produces a unique and clear output, providing technical information
#            or capable of recreating the object, for the "developer". The `repr()`
#            call, typing a variable name and pressing Enter in the console, or
#            debuggers use this method. Although not a rule, a good __repr__ output
#            should be valid Python code that can create a copy of the object when
#            executed as `eval(repr(obj))`.

print("\n--- B) __str__ ve __repr__ Usage (AI/ML Example) ---")

class TrainingConfig:
    def __init__(self, lr, epochs, optimizer='adam'):
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer

    def __str__(self):
        # Readable, "nice" summary for reporting to the user
        return f"Training Configuration (lr={self.lr}, epoch={self.epochs}, optimizer='{self.optimizer}')"

    def __repr__(self):
        # Technical and clear information needed to reproduce the experiment
        return f"TrainingConfig(lr={self.lr}, epochs={self.epochs}, optimizer='{self.optimizer}')"

config = TrainingConfig(lr=0.001, epochs=50)

# 1. The print() function prefers the __str__ method:
print("print(config) output (__str__ is used):")
print(config)

# 2. The repr() function uses the __repr__ method:
print("\nrepr(config) output:")
print(repr(config))

# 3. Collections like lists use the __repr__ representation of their elements:
print("\n[config] output (list uses __repr__):")
print([config])


#--------------------------------------------------------------------------
### C) __slots__: Memory Optimization

# Normally, Python objects store their attributes in a dictionary called __dict__.
# This provides flexibility (you can add new attributes at runtime) but comes with a memory cost.

# __slots__: When this attribute is defined in a class, it tells Python not to
#            create a __dict__ and to only allocate memory for the attributes
#            specified in __slots__. This significantly reduces memory usage,
#            especially when creating tens of thousands or millions of small objects
#            (E.g., millions of feature vectors in a data set, token objects in a
#            large text corpus, pixel objects in an image, or particles in simulations).

# Trade-off: You cannot dynamically assign a new attribute to objects that use __slots__,
#            other than those defined in the slots list (e.g., `object.new_attribute = value`).
#            This is not an error but a conscious limitation imposed by this optimization.

print("\n--- C) __slots__ with Memory Optimization (AI/ML Example) ---")

class FeatureVector_Dict:
    # Consumes a lot of memory in millions of instances
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class FeatureVector_Slots:
    # Much more memory efficient
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# 1. Standard Class (with dict)
fv_dict = FeatureVector_Dict(0.1, 0.2, 0.3)
print(f"Standard vector has a __dict__: {fv_dict.__dict__}")
fv_dict.label = 'cat' # We can add new attributes at runtime.
print(f"New attribute added: fv_dict.label = '{fv_dict.label}'")


# 2. Optimized Class (with slots)
fv_slots = FeatureVector_Slots(0.1, 0.2, 0.3)

# Let's see that there is no __dict__:
try:
    print(fv_slots.__dict__)
except AttributeError as e:
    print(f"\nSlots vector does not have a __dict__. Error: {e}")

# Let's try to add a new attribute:
try:
    fv_slots.label = 'cat'
except AttributeError as e:
    print(f"New attribute cannot be added to slots vector. Error: {e}")
