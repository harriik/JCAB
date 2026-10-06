import json
from modules.assessment import QUESTION_BANK

# Hand-crafted high-quality semantic concepts and keywords for all 50 topics
KNOWLEDGE_BASE = {
    "OOP Fundamentals": {
        "concepts": ["encapsulation", "inheritance", "polymorphism", "abstraction", "objects", "classes", "maintainability", "modularity", "reusability"],
        "keywords": ["encapsulation", "inheritance", "polymorphism", "abstraction", "object", "class", "maintain", "reuse"],
        "explanation": "Object-Oriented Programming (OOP) in Java relies on encapsulation for data hiding, inheritance for reusability, polymorphism for flexibility, and abstraction to simplify complexity. These concepts collectively enable modularity and maintainable code design."
    },
    "Inheritance": {
        "concepts": ["reusability", "extends", "parent class", "child class", "superclass", "subclass", "method overriding", "is-a relationship", "single inheritance", "multilevel inheritance", "multiple inheritance not supported"],
        "keywords": ["extend", "parent", "child", "super", "sub", "override", "reuse", "single", "multilevel"],
        "explanation": "Inheritance allows a child class to acquire properties and methods of a parent class using the 'extends' keyword. It promotes code reusability and establishes an 'is-a' relationship. Java supports single and multilevel inheritance but not multiple inheritance to avoid the diamond problem."
    },
    "Polymorphism": {
        "concepts": ["many forms", "method overriding", "method overloading", "runtime polymorphism", "compile-time polymorphism", "dynamic method dispatch", "interfaces", "upcasting"],
        "keywords": ["form", "override", "overload", "runtime", "compile", "dynamic", "dispatch", "interface"],
        "explanation": "Polymorphism allows objects to be treated as instances of their parent class or interface, taking 'many forms'. It includes compile-time polymorphism through method overloading and runtime polymorphism through method overriding and dynamic method dispatch."
    },
    "Encapsulation": {
        "concepts": ["data hiding", "private fields", "public getters", "public setters", "access modifiers", "security", "control access", "protection", "bundling data and methods"],
        "keywords": ["hide", "private", "getter", "setter", "modifier", "secure", "protect", "bundle"],
        "explanation": "Encapsulation bundles data (variables) and methods operating on that data within a single unit (class). It is achieved by declaring fields as private and providing public getters and setters, enforcing data hiding, access control, and security."
    },
    "Exception Handling": {
        "concepts": ["checked exceptions", "unchecked exceptions", "compile time", "runtime", "try catch finally", "throws keyword", "throw keyword", "custom exceptions", "recovery"],
        "keywords": ["check", "uncheck", "compile", "runtime", "try", "catch", "finally", "throws", "throw", "custom", "recover"],
        "explanation": "Exception handling manages runtime errors to maintain normal application flow. Checked exceptions are verified at compile-time (e.g., IOException), while unchecked exceptions occur at runtime (e.g., NullPointerException). Custom exceptions handle domain-specific errors."
    },
    "Multithreading": {
        "concepts": ["thread class", "runnable interface", "concurrent execution", "start method", "run method", "multiple inheritance limitation", "parallelism", "cpu utilization"],
        "keywords": ["thread", "runnable", "concurrent", "start", "run", "extend", "implement", "parallel"],
        "explanation": "Multithreading enables concurrent execution of two or more parts of a program. A thread is created by extending the Thread class or implementing the Runnable interface. Implementing Runnable is preferred as it avoids the single inheritance limitation."
    },
    "Collections Framework": {
        "concepts": ["list", "set", "map", "ordered collection", "duplicates allowed", "unique elements", "key value pairs", "hashmap", "treemap", "sorted", "hashing"],
        "keywords": ["list", "set", "map", "order", "duplicate", "unique", "key", "value", "hash", "tree", "sort"],
        "explanation": "The Collections Framework provides architectures to store and manipulate groups of objects. A List is ordered and allows duplicates. A Set contains unique elements. A Map stores key-value pairs. HashMap provides O(1) unordered access, while TreeMap provides sorted order."
    },
    "JVM Architecture": {
        "concepts": ["platform independence", "write once run anywhere", "bytecode", "classloader", "memory areas", "heap", "stack", "method area", "execution engine", "interpreter", "jit compiler"],
        "keywords": ["platform", "independent", "wora", "bytecode", "classloader", "memory", "heap", "stack", "jit", "compile", "execute"],
        "explanation": "The JVM (Java Virtual Machine) executes Java bytecode, providing platform independence (Write Once, Run Anywhere). Its architecture includes the ClassLoader, memory areas (Heap, Stack, Method Area), and the Execution Engine featuring an interpreter and JIT compiler."
    },
    "Garbage Collection": {
        "concepts": ["automatic memory management", "unreachable objects", "heap memory", "mark and sweep", "finalize method", "system.gc", "memory leaks", "performance overhead"],
        "keywords": ["memory", "manage", "unreachable", "heap", "mark", "sweep", "finalize", "gc", "leak"],
        "explanation": "Garbage Collection is Java's automatic memory management that identifies and discards unreachable objects to free heap memory. The finalize() method is discouraged due to unpredictability and performance issues; modern alternatives like try-with-resources are preferred."
    },
    "Interfaces vs Abstract Classes": {
        "concepts": ["multiple inheritance", "default methods", "static methods", "abstract methods", "state", "instance variables", "constructors", "is-a", "can-do"],
        "keywords": ["multiple", "inherit", "default", "static", "abstract", "state", "variable", "constructor", "implement", "extend"],
        "explanation": "Abstract classes can hold state (instance variables) and constructors, representing an 'is-a' relationship. Interfaces define a contract ('can-do'), cannot hold instance state, but support multiple inheritance and can contain default and static methods."
    },
    "String Pool": {
        "concepts": ["memory optimization", "heap memory", "immutable", "string literal", "interning", "thread safe", "security", "caching", "performance"],
        "keywords": ["memory", "heap", "immutable", "literal", "intern", "thread", "safe", "cache", "performance"],
        "explanation": "The String Pool is a special storage area in the Java heap. Because strings are immutable (cannot be changed after creation), the JVM caches string literals in the pool to save memory and improve performance, allowing multiple references to point to the same object."
    },
    "Generics": {
        "concepts": ["type safety", "compile-time checking", "type casting", "type erasure", "reusability", "collections", "parameterized types", "backward compatibility"],
        "keywords": ["type", "safe", "compile", "cast", "erasure", "reuse", "collection", "parameter", "compatible"],
        "explanation": "Generics provide compile-time type safety and eliminate the need for explicit type casting. Type erasure is a process where the Java compiler removes generic type information at compile-time to maintain backward compatibility with older Java versions."
    },
    "Lambda Expressions": {
        "concepts": ["functional interface", "anonymous method", "concise syntax", "arrow operator", "single abstract method", "sam", "stream api", "functional programming"],
        "keywords": ["functional", "interface", "anonymous", "concise", "arrow", "single", "abstract", "sam", "stream"],
        "explanation": "Lambda expressions provide a clear and concise syntax to implement Functional Interfaces (interfaces with a Single Abstract Method). They enable functional programming features in Java and are heavily used with the Stream API."
    },
    "Stream API": {
        "concepts": ["functional style", "declarative", "collections processing", "intermediate operations", "terminal operations", "lazy evaluation", "filter", "map", "reduce", "collect"],
        "keywords": ["functional", "declare", "collection", "process", "intermediate", "terminal", "lazy", "filter", "map", "reduce"],
        "explanation": "The Stream API enables functional-style, declarative operations on collections. Intermediate operations (like filter, map) return a new stream and are lazily evaluated. Terminal operations (like collect, reduce) trigger the processing pipeline and produce a result."
    },
    "Optional Class": {
        "concepts": ["nullpointerexception", "npe", "container object", "value presence", "avoid null checks", "functional methods", "ofnullable", "ifpresent", "orelse"],
        "keywords": ["nullpointerexception", "npe", "container", "presence", "null", "check", "functional", "present"],
        "explanation": "The Optional class is a container object used to represent the presence or absence of a value. It was introduced to avoid verbose null checks and prevent NullPointerExceptions by providing functional methods like ifPresent() and orElse()."
    },
    "Concurrency vs Parallelism": {
        "concepts": ["concurrency", "parallelism", "multiple tasks making progress", "tasks executing simultaneously", "threads", "executors", "multiple processor cores", "fork join framework", "java.util.concurrent", "subtasks", "interleaving"],
        "keywords": ["concurren", "parallel", "task", "progress", "simultaneous", "thread", "executor", "core", "fork", "join"],
        "explanation": "Concurrency is about structure—managing multiple tasks making progress at the same time (interleaving). Parallelism is about execution—running tasks simultaneously on multiple processor cores. The Fork/Join framework divides large tasks into subtasks to execute them in parallel across cores."
    },
    "Synchronization": {
        "concepts": ["thread safety", "mutual exclusion", "monitor lock", "race condition", "synchronized block", "synchronized method", "deadlock", "resource contention", "circular wait"],
        "keywords": ["thread", "safe", "mutual", "monitor", "lock", "race", "synchronize", "deadlock", "resource"],
        "explanation": "The 'synchronized' keyword ensures thread safety by allowing only one thread to access a block or method at a time using a monitor lock, preventing race conditions. Deadlock occurs when threads wait on each other cyclically; it can be prevented by ordered lock acquisition."
    },
    "Volatile Keyword": {
        "concepts": ["visibility", "main memory", "cpu cache", "thread safety", "atomicity", "happens-before", "synchronization alternative"],
        "keywords": ["visible", "main", "memory", "cache", "thread", "safe", "atomic", "synchronize"],
        "explanation": "The 'volatile' keyword ensures visibility of variable changes across threads by forcing reads and writes directly to main memory rather than CPU caches. Unlike 'synchronized', it does not provide atomicity or mutual exclusion."
    },
    "Static Keyword": {
        "concepts": ["class level", "memory allocation", "instance independent", "static variables", "static methods", "utility methods", "object creation", "this keyword"],
        "keywords": ["class", "memory", "instance", "independent", "variable", "method", "utility", "object"],
        "explanation": "The 'static' keyword binds a variable or method to the class itself rather than to instances. Static methods cannot directly access non-static instance variables because they run without an object context and have no 'this' reference."
    },
    "Final Keyword": {
        "concepts": ["immutability", "constants", "prevent overriding", "prevent inheritance", "final variable", "final method", "final class", "security", "design"],
        "keywords": ["immutable", "constant", "prevent", "override", "inherit", "variable", "method", "class", "secure"],
        "explanation": "The 'final' keyword restricts modification. A final variable becomes a constant. A final method cannot be overridden by subclasses. A final class cannot be inherited, which is useful for security and immutable designs like the String class."
    },
    "Constructors": {
        "concepts": ["object initialization", "no return type", "same name as class", "default constructor", "parameterized constructor", "constructor overloading", "this keyword", "constructor chaining"],
        "keywords": ["object", "initialize", "return", "name", "default", "parameter", "overload", "chain", "this"],
        "explanation": "A constructor is a special block of code that initializes a newly created object. Constructor overloading provides multiple ways to instantiate a class. The 'this()' keyword is used to call another constructor in the same class (constructor chaining)."
    },
    "Method Overloading vs Overriding": {
        "concepts": ["compile-time polymorphism", "runtime polymorphism", "same method name", "different parameters", "same signature", "inheritance", "static binding", "dynamic binding", "private methods"],
        "keywords": ["compile", "runtime", "polymorph", "name", "parameter", "signature", "inherit", "bind", "private"],
        "explanation": "Overloading occurs in the same class with different parameters (compile-time polymorphism). Overriding occurs in a subclass with the same signature (runtime polymorphism). Private methods cannot be overridden because they are not visible to subclasses."
    },
    "Super Keyword": {
        "concepts": ["parent class", "superclass reference", "invoke parent constructor", "access hidden fields", "override methods", "inheritance hierarchy", "constructor chaining"],
        "keywords": ["parent", "super", "reference", "invoke", "constructor", "hide", "override", "inherit", "chain"],
        "explanation": "The 'super' keyword refers to the immediate parent class object. It is used to access hidden fields, invoke overridden parent methods, and, via 'super()', explicitly call a parent class constructor during subclass instantiation."
    },
    "Wrapper Classes": {
        "concepts": ["primitive types", "object representation", "collections", "generics", "autoboxing", "unboxing", "automatic conversion", "utility methods"],
        "keywords": ["primitive", "object", "collection", "generic", "autobox", "unbox", "automatic", "convert", "utility"],
        "explanation": "Wrapper classes convert primitive data types into objects, required for Java Collections and Generics. Autoboxing is the automatic conversion of a primitive to its wrapper object, while unboxing is the reverse process."
    },
    "Serialization": {
        "concepts": ["byte stream", "network transmission", "file storage", "serializable interface", "transient keyword", "sensitive data", "deserialization", "object state"],
        "keywords": ["byte", "stream", "network", "file", "serial", "transient", "sensitive", "deserial", "state"],
        "explanation": "Serialization converts an object's state into a byte stream for storage or network transmission. The 'transient' keyword marks fields (like passwords) that should be ignored during serialization to protect sensitive data or skip non-serializable references."
    },
    "Cloning": {
        "concepts": ["object copy", "cloneable interface", "clone method", "shallow copy", "deep copy", "reference copy", "independent objects", "mutable fields"],
        "keywords": ["copy", "clone", "shallow", "deep", "reference", "independent", "mutable"],
        "explanation": "Cloning creates a copy of an object. A shallow copy copies primitive values but shares references for mutable objects. A deep copy creates entirely new copies of all referenced mutable objects, ensuring total independence."
    },
    "ClassLoaders": {
        "concepts": ["jvm subsystem", "dynamic loading", "bootstrap classloader", "extension classloader", "application classloader", "system classloader", "delegation model", "bytecode"],
        "keywords": ["jvm", "dynamic", "load", "bootstrap", "extension", "application", "system", "delegate", "bytecode"],
        "explanation": "A ClassLoader is a JVM subsystem that dynamically loads Java classes into memory. The JVM uses three primary built-in loaders: the Bootstrap ClassLoader (core libraries), the Extension ClassLoader, and the Application/System ClassLoader (classpath files)."
    },
    "Reflection API": {
        "concepts": ["runtime inspection", "metadata", "dynamic invocation", "private access", "frameworks", "spring", "hibernate", "performance overhead", "security risks"],
        "keywords": ["runtime", "inspect", "metadata", "dynamic", "invoke", "private", "framework", "performance", "secure"],
        "explanation": "Reflection allows inspection and modification of classes, methods, and fields at runtime, enabling frameworks like Spring to instantiate objects dynamically. However, it incurs performance overhead and bypasses security/access checks (like accessing private fields)."
    },
    "Annotations": {
        "concepts": ["metadata", "compiler instructions", "runtime processing", "built-in annotations", "override", "deprecated", "custom annotations", "retention policy", "target"],
        "keywords": ["metadata", "compiler", "runtime", "built", "override", "deprecate", "custom", "retention", "target"],
        "explanation": "Annotations provide metadata about the code. Built-in annotations (@Override, @Deprecated) give instructions to the compiler. Custom annotations are defined by developers and processed at compile-time or runtime using reflection."
    },
    "Enums": {
        "concepts": ["named constants", "type safety", "singleton", "fields", "methods", "constructors", "switch statements", "iteration"],
        "keywords": ["constant", "type", "safe", "singleton", "field", "method", "constructor", "switch", "iterate"],
        "explanation": "An Enum defines a fixed set of named constants, providing type safety. Unlike simple constants, Java enums are full classes; they can have custom fields, methods, and private constructors to initialize state for each constant."
    },
    "Memory Leaks": {
        "concepts": ["unreachable objects", "garbage collector", "unintended references", "static collections", "unclosed resources", "listeners", "caches", "out of memory"],
        "keywords": ["unreachable", "garbage", "reference", "static", "collection", "resource", "listener", "cache", "memory"],
        "explanation": "Memory leaks occur in Java when the Garbage Collector cannot free objects because unintended strong references still point to them. Common causes include static collections growing indefinitely, unclosed streams, or forgotten event listeners."
    },
    "Comparable vs Comparator": {
        "concepts": ["sorting", "natural ordering", "comparable interface", "compareto method", "custom ordering", "comparator interface", "compare method", "multiple sort sequences", "modifying source"],
        "keywords": ["sort", "natural", "order", "comparable", "compare", "custom", "comparator", "multiple", "modify"],
        "explanation": "Comparable (compareTo) defines the default, natural sorting order within the class itself. Comparator (compare) is implemented externally to provide multiple, custom sorting sequences without modifying the original class source code."
    },
    "Hashcode and Equals": {
        "concepts": ["object equality", "hash based collections", "hashmap", "hashset", "contract", "buckets", "memory address", "collisions", "data retrieval"],
        "keywords": ["equal", "hash", "collection", "contract", "bucket", "memory", "collision", "retrieve"],
        "explanation": "The contract states that if two objects are equal according to equals(), they must have the same hashCode(). If you override equals() but not hashCode(), equal objects will hash to different buckets, breaking collections like HashMap and HashSet."
    },
    "Default Methods": {
        "concepts": ["interfaces", "backward compatibility", "method implementation", "multiple inheritance", "diamond problem", "override", "super keyword"],
        "keywords": ["interface", "backward", "compatible", "implement", "multiple", "inherit", "diamond", "override"],
        "explanation": "Default methods allow interfaces to have concrete implementations, providing backward compatibility for existing interfaces (like Collection). If a class inherits conflicting default methods (the diamond problem), it must explicitly override the method and specify which to use."
    },
    "Type Inference": {
        "concepts": ["compiler deduction", "var keyword", "local variable type inference", "java 10", "readability", "diamond operator", "compile time"],
        "keywords": ["compile", "deduce", "var", "local", "variable", "infer", "read", "diamond", "time"],
        "explanation": "Type inference allows the compiler to deduce the type of a variable from its context. The 'var' keyword (Java 10) enables local variable type inference, reducing verbosity and improving readability without sacrificing compile-time static typing."
    },
    "Records": {
        "concepts": ["data carrier", "immutable", "boilerplate", "pojo", "implicit constructor", "getters", "equals", "hashcode", "tostring", "java 14"],
        "keywords": ["data", "immutable", "boilerplate", "pojo", "implicit", "constructor", "getter", "equal", "hash", "string"],
        "explanation": "Records (Java 14+) are immutable data carrier classes that automatically generate boilerplate code (constructors, getters, equals, hashCode, toString). This drastically reduces the verbosity compared to traditional POJOs."
    },
    "Sealed Classes": {
        "concepts": ["inheritance control", "domain modeling", "permits keyword", "exhaustiveness", "pattern matching", "security", "api design"],
        "keywords": ["inherit", "control", "domain", "model", "permit", "exhaustive", "pattern", "match", "secure", "api"],
        "explanation": "Sealed classes restrict which other classes or interfaces may extend or implement them using the 'permits' keyword. This provides strict control over inheritance hierarchies, improves API design security, and enables exhaustive pattern matching."
    },
    "Pattern Matching": {
        "concepts": ["instanceof", "type casting", "readability", "boilerplate reduction", "scope", "binding variable", "switch expressions", "safety"],
        "keywords": ["instanceof", "cast", "read", "boilerplate", "scope", "bind", "variable", "switch", "safe"],
        "explanation": "Pattern matching for 'instanceof' simplifies type checking and casting by extracting the object into a binding variable in one step. This eliminates redundant cast boilerplate, improving code readability and type safety."
    },
    "JDBC": {
        "concepts": ["database connectivity", "sql execution", "driver manager", "connection", "statement", "preparedstatement", "sql injection", "precompiled", "performance"],
        "keywords": ["database", "sql", "driver", "connect", "statement", "prepare", "inject", "precompile", "performance"],
        "explanation": "JDBC enables Java applications to connect and execute SQL on databases. PreparedStatement is preferred over Statement because it is precompiled (improving performance) and uses parameterized queries, which prevents SQL injection attacks."
    },
    "Design Patterns (Singleton)": {
        "concepts": ["creational pattern", "single instance", "global access", "private constructor", "static method", "thread safe", "double checked locking", "enum singleton"],
        "keywords": ["create", "single", "instance", "global", "access", "private", "constructor", "static", "thread", "safe", "lock", "enum"],
        "explanation": "The Singleton pattern ensures a class has only one instance and provides a global point of access. In Java, thread safety can be achieved using double-checked locking with the volatile keyword, or simply by using an Enum."
    },
    "Design Patterns (Factory)": {
        "concepts": ["creational pattern", "object creation", "hide instantiation logic", "interfaces", "subclasses", "loose coupling", "flexibility"],
        "keywords": ["create", "object", "hide", "instantiate", "interface", "subclass", "loose", "couple", "flexible"],
        "explanation": "The Factory pattern provides an interface for creating objects in a superclass, allowing subclasses to alter the type of objects created. It is used over simple constructors to hide instantiation logic and promote loose coupling."
    },
    "Design Patterns (Observer)": {
        "concepts": ["behavioral pattern", "one-to-many relationship", "publish subscribe", "subject", "state change", "event listeners", "gui", "mvc"],
        "keywords": ["behavior", "relationship", "publish", "subscribe", "subject", "state", "event", "listener", "gui", "mvc"],
        "explanation": "The Observer pattern defines a one-to-many dependency where state changes in a Subject automatically notify multiple Observers. Real-world examples include GUI event listeners (button clicks) or Model-View-Controller (MVC) architectures."
    },
    "SOLID Principles": {
        "concepts": ["single responsibility", "open closed", "liskov substitution", "interface segregation", "dependency inversion", "maintainability", "one reason to change", "cohesion"],
        "keywords": ["single", "responsible", "open", "close", "liskov", "substitute", "interface", "segregate", "dependency", "invert", "maintain"],
        "explanation": "SOLID principles promote maintainable object-oriented design. The Single Responsibility Principle (SRP) dictates that a class should have only one reason to change, meaning it should encapsulate exactly one focused responsibility or functionality."
    },
    "Dependency Injection": {
        "concepts": ["inversion of control", "ioc", "loose coupling", "testability", "constructor injection", "setter injection", "spring framework", "external configuration"],
        "keywords": ["invert", "control", "ioc", "loose", "couple", "test", "constructor", "inject", "setter", "spring", "external"],
        "explanation": "Dependency Injection (DI) is a technique where an object receives its dependencies from an external source rather than creating them itself. This promotes loose coupling, making code easier to test, maintain, and configure."
    },
    "Unit Testing": {
        "concepts": ["junit", "test isolation", "code quality", "regression", "mocking", "mockito", "stubbing", "dependencies", "behavior verification"],
        "keywords": ["junit", "test", "isolate", "quality", "regression", "mock", "mockito", "stub", "dependency", "behavior"],
        "explanation": "Unit testing verifies individual components in isolation to ensure code quality and prevent regressions. Mocking frameworks like Mockito create fake objects (mocks) for external dependencies, allowing developers to test logic without touching actual databases or APIs."
    },
    "Build Tools": {
        "concepts": ["maven", "gradle", "automation", "dependency management", "pom.xml", "transitive dependencies", "repositories", "build lifecycle", "compile package deploy"],
        "keywords": ["maven", "gradle", "automate", "dependency", "manage", "pom", "transitive", "repository", "build", "lifecycle"],
        "explanation": "Build tools like Maven automate compilation, testing, and packaging. Maven's dependency management handles fetching external libraries (and their transitive dependencies) via the pom.xml file, ensuring consistent builds across environments."
    },
    "REST APIs": {
        "concepts": ["representational state transfer", "http protocol", "stateless", "json", "endpoints", "get", "post", "put", "delete", "spring boot", "jax-rs"],
        "keywords": ["represent", "state", "transfer", "http", "stateless", "json", "endpoint", "get", "post", "put", "delete", "spring"],
        "explanation": "REST APIs expose resources over HTTP using standard methods in a stateless manner. Common HTTP methods include GET (retrieve), POST (create), PUT (update), and DELETE (remove). Java commonly implements them using Spring Boot or JAX-RS."
    },
    "Spring Framework": {
        "concepts": ["enterprise java", "inversion of control", "ioc container", "dependency injection", "application context", "beans", "loose coupling", "spring boot", "auto-configuration"],
        "keywords": ["enterprise", "java", "invert", "control", "ioc", "container", "dependency", "inject", "context", "bean", "loose"],
        "explanation": "Spring is a comprehensive enterprise framework. Inversion of Control (IoC) means the Spring container takes control of creating and managing objects (Beans) and injecting their dependencies, removing that responsibility from the application code."
    },
    "Microservices": {
        "concepts": ["architectural style", "independent deployment", "domain driven design", "scalability", "spring boot", "distributed systems", "complexity", "network latency", "api gateway"],
        "keywords": ["architecture", "independent", "deploy", "domain", "scale", "spring", "distribute", "complex", "network", "api"],
        "explanation": "Microservices architecture breaks down a monolith into small, independently deployable services around business domains. Benefits include targeted scalability and technology flexibility, while challenges include distributed system complexity and network latency."
    },
    "JPA / Hibernate": {
        "concepts": ["java persistence api", "orm", "object relational mapping", "hibernate", "entity manager", "transient state", "persistent state", "detached state", "database sync"],
        "keywords": ["java", "persist", "api", "orm", "object", "relation", "map", "hibernate", "entity", "transient", "persistent", "detach"],
        "explanation": "JPA is the specification for ORM (Object-Relational Mapping), and Hibernate is its popular implementation. A transient entity is new and unmanaged; a persistent entity is managed by the EntityManager and synced to the DB; a detached entity was managed but its session closed."
    }
}

rubrics = {}
for i, (topic, (q1, q2)) in enumerate(QUESTION_BANK.items(), 1):
    qid = f"Q_{i:03d}"
    
    # Use hand-crafted knowledge if available, otherwise fallback
    kb = KNOWLEDGE_BASE.get(topic, {})
    
    expected_concepts = kb.get("concepts", [topic.lower(), "java", "programming"])
    important_keywords = kb.get("keywords", [topic.lower().split()[0], "class", "method"])
    reference_explanation = kb.get("explanation", f"This concept relates to {topic}. It is a fundamental part of Java programming.")
    
    rubrics[topic] = {
        "question_id": qid,
        "topic": topic,
        "expected_concepts": expected_concepts,
        "important_keywords": important_keywords,
        "reference_explanation": reference_explanation
    }

with open("data/java_fallback_rubrics.json", "w", encoding="utf-8") as f:
    json.dump(rubrics, f, indent=2)

print("Generated comprehensive rubrics for all 50 questions!")
