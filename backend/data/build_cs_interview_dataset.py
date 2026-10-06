"""Build a reproducible, structured Computer Science interview question bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "cs_interview_dataset.jsonl"
ROLE_PATH = DATA_DIR / "cs_interview_job_roles.json"

ROLE_SKILLS: dict[str, list[str]] = {
    "Software Developer": ["programming", "data structures", "algorithms", "oop", "software engineering", "testing", "git", "system design"],
    "Python Developer": ["python", "programming", "data structures", "algorithms", "oop", "api", "testing", "git"],
    "Java Developer": ["java", "programming", "data structures", "algorithms", "oop", "dbms", "api", "testing", "git"],
    "Full Stack Developer": ["javascript", "html", "css", "web development", "frontend", "backend", "api", "dbms", "git", "testing"],
    "Frontend Developer": ["javascript", "html", "css", "web development", "frontend", "api", "testing", "git"],
    "Backend Developer": ["programming", "backend", "api", "dbms", "system design", "distributed systems", "security", "testing", "git"],
    "Data Analyst": ["sql", "dbms", "statistics", "data science", "programming", "cloud"],
    "Data Scientist": ["python", "programming", "data science", "machine learning", "statistics", "ai"],
    "AI Engineer": ["programming", "ai", "machine learning", "deep learning", "nlp", "generative ai", "llm", "api", "cloud"],
    "ML Engineer": ["python", "programming", "machine learning", "deep learning", "data science", "cloud", "devops"],
    "Generative AI Engineer": ["python", "programming", "generative ai", "llm", "nlp", "machine learning", "api", "cloud"],
    "Computer Vision Engineer": ["python", "programming", "computer vision", "deep learning", "machine learning", "data science"],
    "DevOps Engineer": ["linux", "networking", "cloud", "devops", "distributed systems", "security", "git", "testing"],
    "Cloud Engineer": ["networking", "cloud", "devops", "distributed systems", "security", "system design"],
    "Cybersecurity Engineer": ["security", "networking", "operating systems", "programming", "cloud"],
    "QA Engineer": ["testing", "software engineering", "programming", "api", "web development", "git"],
    "Blockchain Developer": ["blockchain", "programming", "security", "distributed systems", "api", "git"],
}

ROLE_SUBJECTS: dict[str, list[str]] = {
    "Programming": ["Software Developer", "Python Developer", "Java Developer", "Backend Developer", "Data Scientist", "AI Engineer", "ML Engineer", "Generative AI Engineer", "Computer Vision Engineer", "Cybersecurity Engineer", "QA Engineer", "Blockchain Developer"],
    "Data Structures": ["Software Developer", "Python Developer", "Java Developer"],
    "Algorithms": ["Software Developer", "Python Developer", "Java Developer", "Data Scientist", "AI Engineer", "ML Engineer"],
    "OOP": ["Software Developer", "Python Developer", "Java Developer", "Backend Developer"],
    "DBMS": ["Java Developer", "Full Stack Developer", "Backend Developer", "Data Analyst", "Data Scientist"],
    "Operating Systems": ["DevOps Engineer", "Cybersecurity Engineer"],
    "Computer Networks": ["Backend Developer", "DevOps Engineer", "Cloud Engineer", "Cybersecurity Engineer"],
    "Software Engineering": ["Software Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer", "QA Engineer"],
    "Computer Architecture": ["Software Developer", "AI Engineer", "ML Engineer"],
    "Web Development": ["Full Stack Developer", "Frontend Developer", "Backend Developer", "QA Engineer"],
    "Artificial Intelligence": ["Data Scientist", "AI Engineer", "ML Engineer"],
    "Machine Learning": ["Data Scientist", "AI Engineer", "ML Engineer", "Generative AI Engineer", "Computer Vision Engineer"],
    "Deep Learning": ["AI Engineer", "ML Engineer", "Generative AI Engineer", "Computer Vision Engineer"],
    "NLP": ["AI Engineer", "Generative AI Engineer"],
    "Generative AI": ["AI Engineer", "Generative AI Engineer"],
    "LLM": ["AI Engineer", "Generative AI Engineer"],
    "Computer Vision": ["AI Engineer", "Computer Vision Engineer"],
    "Cybersecurity": ["Backend Developer", "DevOps Engineer", "Cloud Engineer", "Cybersecurity Engineer", "Blockchain Developer"],
    "Cloud Computing": ["Backend Developer", "Data Analyst", "AI Engineer", "ML Engineer", "Generative AI Engineer", "DevOps Engineer", "Cloud Engineer"],
    "DevOps": ["Backend Developer", "ML Engineer", "DevOps Engineer", "Cloud Engineer"],
    "System Design": ["Software Developer", "Backend Developer", "Cloud Engineer"],
    "Distributed Systems": ["Backend Developer", "AI Engineer", "DevOps Engineer", "Cloud Engineer", "Blockchain Developer"],
    "Blockchain": ["Blockchain Developer"],
    "Data Science": ["Data Analyst", "Data Scientist", "ML Engineer"],
    "Software Testing": ["Software Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer", "QA Engineer"],
    "Git and GitHub": ["Software Developer", "Python Developer", "Java Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer", "QA Engineer", "DevOps Engineer", "Blockchain Developer"],
    "API and Web Services": ["Java Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer", "AI Engineer", "Generative AI Engineer", "QA Engineer", "Blockchain Developer"],
    "IoT": ["Software Developer", "Cloud Engineer"],
    "Theory of Computation": ["Software Developer"],
    "Compiler Design": ["Software Developer"],
}

EXTRA_TOPICS: list[dict[str, Any]] = [
    {
        "subject": "Programming",
        "topic": "exception handling",
        "skills": ["programming", "python", "java"],
        "job_roles": ["Software Developer", "Python Developer", "Java Developer", "Backend Developer", "QA Engineer"],
        "question_type": "definition",
        "question": "What is an exception, and how does exception handling help a program?",
        "simple_answer": "An exception is an event raised while a program runs that interrupts the current operation. A try/except or equivalent handler can recover, report the problem, or clean up safely.",
        "book_definition": "An exception is a runtime signal representing an abnormal condition or failure that changes the ordinary control flow. Exception-handling constructs transfer control to a matching handler so a program can recover, report the failure, or perform cleanup.",
        "example": "A program can catch FileNotFoundError when opening a missing file, explain how to provide the file, and close any resources it already opened.",
        "reference_answer": "An exception is an event raised during program execution that interrupts the normal control flow. Exception handling uses constructs such as try/except to catch expected failures, recover or report a useful error, and clean up resources. Handlers should be specific; catching every error silently can hide bugs.",
        "key_points": ["runtime event interrupts normal control flow", "exception handling manages errors", "handling can report failures or recover when possible"],
        "common_misconceptions": [{"trigger": "exception handling prevents every program crash", "correction": "Handlers only manage exceptions they catch; unexpected failures can still propagate, and not every failure is safely recoverable."}],
    },
    {
        "subject": "Data Structures",
        "topic": "Python lists",
        "skills": ["data structures", "python", "programming"],
        "job_roles": ["Software Developer", "Python Developer", "Data Scientist", "AI Engineer", "ML Engineer"],
        "question_type": "definition",
        "question": "What is a Python list? Give a simple explanation and a practical example.",
        "simple_answer": "A Python list stores multiple values in an ordered collection that can be changed.",
        "book_definition": "A Python list is a mutable, ordered sequence that stores references to objects and supports indexed access and sequence operations.",
        "example": "A shopping cart can be represented as items = [\"bread\", \"milk\"]; items.append(\"eggs\") adds another item.",
        "reference_answer": "A Python list is a mutable, ordered sequence that stores references to objects.",
        "key_points": ["ordered sequence", "mutable collection", "stores multiple items"],
        "common_misconceptions": [],
    },
    {
        "subject": "Data Structures",
        "topic": "Python list vs tuple",
        "skills": ["data structures", "python", "programming"],
        "job_roles": ["Software Developer", "Python Developer", "Data Scientist", "AI Engineer", "ML Engineer"],
        "question_type": "comparison",
        "question": "Compare a Python list and tuple. Explain their differences, advantages, disadvantages, use cases, and which to choose.",
        "simple_answer": "Lists are mutable; tuples are immutable. Choose a list for a collection that changes and a tuple for fixed grouped values.",
        "book_definition": "A list is a mutable sequence, while a tuple is an immutable sequence. Both preserve order and support indexing, but their mutability gives them different use cases.",
        "example": "Use a list for tasks that are added or removed; use a tuple for a fixed coordinate such as (x, y).",
        "reference_answer": "A Python list is mutable, so items can be added, removed, or replaced. A tuple is immutable after creation, though an object stored inside it may itself be mutable.\n\nAdvantages of a list: flexible updates and many mutating methods. Disadvantages: its contents can change unexpectedly when shared, and it is not hashable.\n\nAdvantages of a tuple: communicates fixed grouped data and can be hashable when all its elements are hashable. Disadvantages: it cannot be resized or have its elements reassigned.\n\nUse a list for a changing collection such as a queue of tasks (with an appropriate queue implementation). Use a tuple for a fixed record such as a coordinate. Neither is always better: choose based on whether the sequence should change.",
        "key_points": ["both are ordered sequences", "list is mutable", "tuple is immutable", "list suits changing collections", "tuple suits fixed grouped values", "choice depends on whether values should change"],
        "common_misconceptions": [{"trigger": "tuples are always hashable", "correction": "A tuple is hashable only when all of its elements are hashable."}],
        "comparison": {"other": "tuple", "a": "A list is an ordered mutable sequence.", "b": "A tuple is an ordered immutable sequence.", "difference": "Lists support item updates and resizing; tuple item bindings cannot be reassigned after construction.", "pros_a": "Flexible; convenient for changing collections.", "cons_a": "Mutable shared state can be changed unexpectedly; lists are not hashable.", "pros_b": "Represents fixed grouped values and may be hashable if its contents are hashable.", "cons_b": "Cannot be resized or have its item bindings reassigned.", "use_a": "Collections expected to change.", "use_b": "Fixed records such as coordinates or return values.", "better": "Use a list when the collection changes and a tuple when it represents fixed grouped values."},
    },
    {
        "subject": "Algorithms",
        "topic": "binary search vs linear search",
        "skills": ["algorithms", "data structures", "programming"],
        "job_roles": ["Software Developer", "Python Developer", "Java Developer", "Data Scientist", "AI Engineer", "ML Engineer"],
        "question_type": "comparison",
        "question": "Compare binary search and linear search. Explain their differences, advantages, disadvantages, use cases, and which is better.",
        "simple_answer": "Linear search checks items one by one and works on unsorted data. Binary search halves a sorted search range and needs O(log n) comparisons.",
        "book_definition": "Linear search examines sequence elements until a match or exhaustion. Binary search repeatedly halves an ordered search interval while preserving the possibility that the target remains in that interval.",
        "example": "For a sorted phone directory, binary search jumps to the middle; for a short unsorted list, linear search checks each item.",
        "reference_answer": "Linear search checks elements one at a time. It works on unsorted data, is simple, and takes O(n) time in the worst case; its disadvantage is that it may inspect every item.\n\nBinary search compares against the middle and discards half the remaining range. It takes O(log n) comparisons, but requires sorted data or a valid monotonic predicate and careful boundary handling. Sorting solely for one lookup may cost more than scanning.\n\nUse linear search for small or unsorted collections and one-off lookups. Use binary search for large, already-sorted data or repeated lookups where ordering is maintained. Binary search is not universally better; choose based on data ordering, collection size, and whether sorting cost is worthwhile.",
        "key_points": ["linear search checks items one by one", "linear search works on unsorted input and is O(n)", "binary search halves an ordered range and is O(log n)", "binary search requires sorted or monotonic input", "linear search suits small or unsorted data", "binary search suits large sorted data or repeated lookups"],
        "common_misconceptions": [{"trigger": "binary search works on any array", "correction": "Binary search requires sorted data or a valid monotonic predicate."}],
        "comparison": {"other": "linear search", "a": "Binary search halves a sorted or monotonic search range.", "b": "Linear search inspects elements sequentially.", "difference": "Binary search takes O(log n) comparisons with its ordering precondition; linear search takes O(n) worst-case time and works without ordering.", "pros_a": "Fewer comparisons on large ordered inputs.", "cons_a": "Requires ordering and correct boundary logic.", "pros_b": "Simple and works on unsorted data.", "cons_b": "May inspect all elements.", "use_a": "Large sorted collections and repeated queries.", "use_b": "Small or unsorted collections and one-off lookups.", "better": "Choose binary search when data is already ordered and large enough; otherwise linear search may be simpler and faster overall."},
    },
    {
        "subject": "Data Structures",
        "topic": "queue vs stack for ticket entry",
        "skills": ["data structures", "programming"],
        "job_roles": ["Software Developer", "Python Developer", "Java Developer", "Backend Developer"],
        "question_type": "comparison",
        "question": "For people entering through a ticket line, should the system use a queue or a stack? Explain why and when the other is appropriate.",
        "simple_answer": "Use a queue because ticket entry normally serves people in arrival order (first in, first out). A stack serves the most recently added item first.",
        "book_definition": "A queue is an abstract data type that inserts at the rear and removes from the front (FIFO). A stack inserts and removes at one end (LIFO).",
        "example": "People waiting at a venue gate are admitted in arrival order, so enqueue each arrival and dequeue the next person to serve.",
        "reference_answer": "Use a queue for a ticket-entry line: it is first in, first out (FIFO), so the person who arrived first is normally served first. This is fair and matches the real-world line. A stack is last in, first out (LIFO), so it would serve the most recent arrival first and is not appropriate for ordinary ticket entry. A stack is useful for undo history or nested function calls. If ticket holders have priority classes or timed slots, use a priority queue or scheduling policy instead of assuming simple arrival order.",
        "key_points": ["queue is first in first out", "stack is last in first out", "ordinary ticket entry serves arrival order", "queue is the appropriate choice", "priority or timed tickets may need another policy"],
        "common_misconceptions": [],
        "comparison": {"other": "stack", "a": "A queue serves the earliest arrival first (FIFO).", "b": "A stack serves the most recent arrival first (LIFO).", "difference": "The removal order differs: queue removes from the front, stack removes from the top.", "pros_a": "Preserves arrival order for fair lines and task scheduling.", "cons_a": "Does not naturally prioritize urgent or reserved entries.", "pros_b": "Efficient for undo operations and nested/recent work.", "cons_b": "Newest entries jump ahead of older ones.", "use_a": "Ticket lines and first-come-first-served work.", "use_b": "Undo history, parsing, and call stacks.", "better": "A queue is better for ordinary ticket entry; use a priority queue if ticket classes override arrival order."},
        "scenario": ("A venue admits ticket holders in arrival order, but VIP ticket holders may enter first. Which structure or policy fits?", "Use a priority queue or separate VIP and general queues with an explicit fairness rule. A plain FIFO queue fits ordinary entry; a stack would reverse arrival order."),
    },
]

# Each record is original instructional material, not copied from an external source.
# Fields: subject, topic, formal/simple answers, example, why/when, comparison, scenario,
# relevant skills, rubric points, common misconception checks, and optional code.
SUBJECTS: list[dict[str, Any]] = [
    {
        "subject": "Programming", "topic": "functions and modular programming", "skills": ["programming", "python", "java"],
        "book": "A function is a named, callable program unit that may accept parameters and return a result; modular programming organizes behavior into cohesive units with explicit interfaces.",
        "simple": "A function is a reusable block of code that does one job.",
        "example": "A calculate_total(items) function can be called by checkout and invoice code instead of repeating the calculation.",
        "why": "Functions reduce duplication, give behavior a clear name, and make code easier to test and maintain.",
        "when": "Use a function when a behavior is reusable, conceptually distinct, or benefits from isolated testing; avoid splitting trivial code into meaningless fragments.",
        "compare": {"other": "inline repeated code", "a": "A function centralizes behavior behind a callable interface.", "b": "Inline copies place the behavior at each use site.", "difference": "A change to a function is made once, while duplicated code must be kept consistent at every copy.", "pros_a": "Reuse, testability, and one place to maintain behavior.", "cons_a": "Poorly chosen boundaries can create indirection or overly broad interfaces.", "pros_b": "The operation is visible locally and may be appropriate for a tiny one-off expression.", "cons_b": "Copies drift and make fixes error-prone.", "use_a": "Repeated or independently testable behavior.", "use_b": "A short, genuinely one-off operation.", "better": "Prefer a function for reusable or meaningful behavior; inline only when it is simpler and truly local."},
        "scenario": ("Two services calculate a discount with duplicated code, and a policy change caused different results. What would you change?", "Extract a well-named discount function or module, define its inputs and edge cases, add focused tests, and have both services call it."),
        "points": ["reusable callable unit", "parameters and return value", "reduces duplication", "easier to test and maintain"],
        "misconceptions": [{"trigger": "functions always make code faster", "correction": "Functions primarily improve structure and reuse; they do not guarantee faster execution."}],
        "code": {"language": "python", "question": "Write a function that returns the square of an integer.", "answer": "def square(value):\n    return value * value", "output": "square(4) returns 16", "time": "O(1)", "space": "O(1)"},
    },
    {
        "subject": "Data Structures", "topic": "hash maps", "skills": ["data structures", "python", "java"],
        "book": "A hash map is an associative data structure that maps keys to values using a hash function and a collision-resolution strategy.",
        "simple": "A hash map stores key-value pairs so a value can usually be found quickly by its key.",
        "example": "A username-to-user-record map can retrieve a record by username without scanning every user.",
        "why": "Hash maps provide expected constant-time lookup and are useful for counting, indexing, caching, and membership-related algorithms.",
        "when": "Use one when fast key-based lookup matters and keys have stable equality and hashing; choose an ordered structure when sorted traversal or range queries are required.",
        "compare": {"other": "balanced search tree", "a": "A hash map uses hashing and does not inherently keep keys sorted.", "b": "A balanced search tree keeps keys ordered by comparisons.", "difference": "Hash lookup is expected O(1); balanced-tree lookup is O(log n) and supports ordered traversal.", "pros_a": "Fast expected point lookup and simple frequency maps.", "cons_a": "No natural ordering and performance depends on hashing and collisions.", "pros_b": "Sorted iteration, range queries, and predictable logarithmic operations.", "cons_b": "More pointer or node overhead and typically slower point operations.", "use_a": "Exact-key lookup when order is unnecessary.", "use_b": "Ordered keys, predecessor/successor, or range-query workloads.", "better": "Neither is universally better: choose a hash map for expected fast exact lookup and a tree for ordering or ranges."},
        "scenario": ("A service repeatedly checks whether millions of IDs have already been processed. What structure and trade-off would you discuss?", "Use a hash set or a hash map keyed by ID for expected O(1) membership checks. Account for memory use, key hashing, and collision behavior; use a disk-backed or partitioned approach if the set does not fit in memory."),
        "points": ["maps keys to values", "uses hashing", "expected constant-time lookup", "collisions and ordering limitations"],
        "misconceptions": [{"trigger": "hash maps guarantee constant time", "correction": "Constant time is expected under suitable hashing and load assumptions, not a worst-case guarantee."}],
        "code": {"language": "python", "question": "Implement a stack with push and pop using a Python list.", "answer": "stack = []\nstack.append(10)  # push\nvalue = stack.pop()  # pop", "output": "After the push, pop returns 10 and the stack is empty.", "time": "O(1) amortized for append and pop at the end", "space": "O(n) for n stored values"},
    },
    {
        "subject": "Algorithms", "topic": "binary search", "skills": ["algorithms", "programming"],
        "book": "Binary search is a comparison-based search procedure that repeatedly halves an ordered search interval while preserving the possibility that the target lies within it.",
        "simple": "Binary search checks the middle of sorted data and discards the half that cannot contain the answer.",
        "example": "To find 7 in [1, 3, 5, 7, 9], inspect the middle value and narrow the interval until index 3 is found.",
        "why": "It reduces the number of comparisons from linear scanning to logarithmic growth when the search space has a valid monotonic ordering.",
        "when": "Use binary search on sorted data or a monotonic predicate when each comparison safely eliminates half the remaining candidates.",
        "compare": {"other": "linear search", "a": "Binary search halves a valid ordered interval.", "b": "Linear search checks elements in sequence without an ordering requirement.", "difference": "Binary search is O(log n) with its precondition; linear search is O(n) and works on unsorted data.", "pros_a": "Far fewer comparisons for large ordered inputs.", "cons_a": "Requires sorted data or a proven monotonic predicate and careful boundary handling.", "pros_b": "Simple, works on arbitrary sequences, and can stop early.", "cons_b": "May inspect every element.", "use_a": "Large sorted collections or monotonic answer spaces.", "use_b": "Small or unsorted data, or one-off searches where sorting is not worthwhile.", "better": "Use binary search when its ordering invariant holds and repeated queries justify it; otherwise linear search is safer or cheaper overall."},
        "scenario": ("A sorted array has one million values and receives many membership queries. How would you search and what must be true?", "Use binary search for each query, giving O(log n) comparisons, provided the array stays sorted and the midpoint/boundary updates preserve the candidate interval. Include tests for empty input and endpoints."),
        "points": ["requires sorted or monotonic search space", "halves the interval", "O(log n) time", "boundary invariant and edge cases"],
        "misconceptions": [{"trigger": "binary search works on any array", "correction": "Ordinary binary search requires ordering or a valid monotonic predicate."}],
        "code": {"language": "python", "question": "Implement iterative binary search on an ascending list; return the index or -1.", "answer": "def binary_search(values, target):\n    low, high = 0, len(values) - 1\n    while low <= high:\n        middle = low + (high - low) // 2\n        if values[middle] == target:\n            return middle\n        if values[middle] < target:\n            low = middle + 1\n        else:\n            high = middle - 1\n    return -1", "output": "binary_search([2, 4, 7, 10], 7) returns 2", "time": "O(log n)", "space": "O(1)"},
    },
    {
        "subject": "OOP", "topic": "composition and inheritance", "skills": ["oop", "programming", "java", "python"],
        "book": "Inheritance is a mechanism for defining a subtype from a base type; composition constructs behavior by holding and collaborating with other objects.",
        "simple": "Inheritance is an “is-a” relationship; composition builds an object using other objects that it “has.”",
        "example": "A Car can have an Engine object (composition); a SavingsAccount may specialize an Account contract (inheritance) when it truly satisfies that contract.",
        "why": "Both mechanisms reuse behavior, but composition usually keeps dependencies replaceable while inheritance expresses substitutability.",
        "when": "Prefer composition for interchangeable collaborators and flexible behavior; use inheritance when a real subtype relationship and substitutable contract exist.",
        "compare": {"other": "composition", "a": "Inheritance derives a type and behavior from a base type.", "b": "Composition delegates behavior to contained collaborators.", "difference": "Inheritance couples a subtype to a hierarchy; composition connects objects through explicit dependencies.", "pros_a": "Can express a stable subtype contract and reuse shared implementation.", "cons_a": "Deep or changing hierarchies are rigid and can violate substitutability.", "pros_b": "Collaborators can be replaced, tested, and combined flexibly.", "cons_b": "May require explicit delegation and more wiring.", "use_a": "A genuine is-a relationship with a stable behavioral contract.", "use_b": "A has-a relationship or behavior that should vary independently.", "better": "Prefer composition for flexibility; choose inheritance when subtype semantics—not just code reuse—are correct."},
        "scenario": ("Several report exporters share formatting behavior, but each uses a different output destination. How would you design them?", "Keep formatting separate from destination-specific writers and compose an exporter with a writer interface. Use inheritance only if exporters form a genuine substitutable subtype family."),
        "points": ["inheritance models subtype relationships", "composition delegates to collaborators", "composition supports flexible replacement", "inheritance requires substitutability"],
        "misconceptions": [{"trigger": "inheritance is always better for code reuse", "correction": "Inheritance is not automatically preferable; composition often reduces coupling when there is no true subtype relationship."}],
        "code": {"language": "python", "question": "Show composition in Python with a Car that delegates starting behavior to an Engine.", "answer": "class Engine:\n    def start(self):\n        return \"started\"\n\nclass Car:\n    def __init__(self, engine):\n        self.engine = engine\n\n    def start(self):\n        return self.engine.start()", "output": "Car(Engine()).start() returns \"started\".", "time": "O(1) for the shown method calls", "space": "O(1) excluding the objects"},
    },
    {
        "subject": "DBMS", "topic": "normalization and denormalization", "skills": ["dbms", "sql"],
        "book": "Database normalization structures relations to reduce redundancy and update anomalies by decomposing data according to dependencies while preserving required information.",
        "simple": "Normalization organizes tables to avoid storing the same fact in many places; denormalization deliberately duplicates or precomputes data for read efficiency.",
        "example": "Store a customer's address once in a customer table; a reporting view may precompute order totals for frequent dashboards.",
        "why": "Normalization protects consistency; denormalization can reduce joins or computation for measured read bottlenecks.",
        "when": "Normalize transactional data by default; denormalize selectively after measuring a workload and adding a plan to keep duplicated values consistent.",
        "compare": {"other": "denormalization", "a": "Normalization separates facts to reduce redundancy and anomalies.", "b": "Denormalization stores derived or repeated values to optimize reads.", "difference": "Normalization favors update consistency; denormalization trades extra storage and synchronization for read performance.", "pros_a": "Less duplication and fewer insert, update, and delete anomalies.", "cons_a": "Queries may need more joins and can be harder to optimize.", "pros_b": "Can simplify or accelerate common read queries.", "cons_b": "Duplicated data can become stale and writes become more complex.", "use_a": "Operational transactional schemas and sources of truth.", "use_b": "Measured read-heavy workloads, reporting, or materialized views.", "better": "Normalize the source of truth; denormalize only for a demonstrated bottleneck with explicit consistency handling."},
        "scenario": ("A dashboard joins several large tables and misses its latency target. Would you denormalize immediately?", "First inspect query plans, indexes, and measurement data. If joins remain the bottleneck, consider a materialized view or derived read model, define refresh/consistency behavior, and monitor storage and write costs."),
        "points": ["normalization reduces redundancy", "prevents update anomalies", "denormalization trades consistency or write cost for reads", "measure before denormalizing"],
        "misconceptions": [{"trigger": "denormalization always makes databases faster", "correction": "It can improve particular reads but can increase write cost, inconsistency risk, and storage."}],
        "code": {"language": "sql", "question": "Write a query listing every department and its employee count, including empty departments.", "answer": "SELECT d.department_id, COUNT(e.employee_id) AS employee_count\nFROM departments AS d\nLEFT JOIN employees AS e ON e.department_id = d.department_id\nGROUP BY d.department_id;", "output": "A department with no employees has employee_count 0.", "time": "Depends on the query plan and indexes", "space": "Depends on the query plan"},
    },
    {
        "subject": "Operating Systems", "topic": "processes and threads", "skills": ["operating systems", "linux"],
        "book": "A process is an executing program instance with an operating-system-managed resource and address-space context; a thread is an execution sequence scheduled within a process.",
        "simple": "A process is a running program with its own resources; threads are execution paths that share a process's memory.",
        "example": "A browser may isolate tabs in processes while using multiple threads inside a process for UI and background work.",
        "why": "Threads allow concurrent work with lower sharing overhead, while process isolation can contain failures and protect address spaces.",
        "when": "Use threads for concurrent tasks that benefit from shared memory or I/O concurrency; use separate processes when isolation, fault containment, or CPU parallelism under runtime constraints is important.",
        "compare": {"other": "processes", "a": "Threads share an address space and many process resources.", "b": "Processes have separate address spaces and stronger isolation.", "difference": "Thread communication can be direct but requires synchronization; processes communicate through explicit IPC and have higher isolation boundaries.", "pros_a": "Lower creation/communication overhead and convenient shared state.", "cons_a": "Races and one faulty thread can affect the process.", "pros_b": "Better fault and memory isolation.", "cons_b": "More resource overhead and explicit communication.", "use_a": "Related concurrent work with controlled shared state.", "use_b": "Independent services, isolation, or fault containment.", "better": "Choose threads for coordinated shared-memory work and processes when isolation is more valuable than communication overhead."},
        "scenario": ("A media application must decode multiple files concurrently, and one decoder may crash. How would you choose a concurrency boundary?", "Use worker processes if crash containment and isolation matter, with a queue or IPC protocol for results. Threads may be preferable if decoding libraries release the runtime lock and shared-memory overhead dominates; measure both."),
        "points": ["process owns resource and address-space context", "threads execute within a process", "threads share memory", "processes offer stronger isolation"],
        "misconceptions": [{"trigger": "threads have separate memory by default", "correction": "Threads in one process normally share its address space, though each has its own stack and execution state."}],
    },
    {
        "subject": "Computer Networks", "topic": "TCP and UDP", "skills": ["networking", "api"],
        "book": "TCP is a connection-oriented transport protocol providing an ordered reliable byte stream; UDP is a connectionless datagram transport that does not itself guarantee delivery or ordering.",
        "simple": "TCP retries and orders data; UDP sends independent packets with less protocol overhead but no built-in delivery guarantee.",
        "example": "A file transfer commonly uses TCP; a real-time voice stream may use UDP and tolerate some packet loss.",
        "why": "Transport choice determines reliability, ordering, latency behavior, and what recovery logic the application must provide.",
        "when": "Choose TCP when complete ordered delivery is required; consider UDP when timeliness matters more than occasional loss and the application can handle it.",
        "compare": {"other": "UDP", "a": "TCP provides a reliable, ordered byte stream with congestion and flow control.", "b": "UDP provides datagrams without transport-level reliability or ordering guarantees.", "difference": "TCP handles retransmission and ordering; UDP leaves those policies to the application.", "pros_a": "Reliable ordered delivery and broad application support.", "cons_a": "Retransmissions and connection behavior can add latency.", "pros_b": "Datagram boundaries and low protocol overhead.", "cons_b": "Loss, duplication, and reordering need application handling if relevant.", "use_a": "Web pages, file transfer, and transactions needing complete ordered data.", "use_b": "Real-time media, telemetry, or protocols that implement their own reliability.", "better": "Neither is universally better: use TCP for reliable streams and UDP when the latency/loss trade-off is intentional."},
        "scenario": ("A live voice call sounds delayed when packets are retransmitted. Which transport behavior may help, and what must the application accept?", "UDP can avoid transport-level retransmission delays, but audio may be lost or arrive out of order. The application should use timing, jitter buffers, concealment, and possibly selective recovery suited to real-time deadlines."),
        "points": ["TCP reliable ordered byte stream", "UDP datagram transport without reliability guarantee", "TCP retransmission can add latency", "choose based on loss and latency needs"],
        "misconceptions": [{"trigger": "udp is always faster", "correction": "UDP has less transport machinery, but actual performance depends on network, protocol, and application behavior."}],
    },
    {
        "subject": "Software Engineering", "topic": "monolith and microservices", "skills": ["software engineering", "system design", "backend"],
        "book": "A monolith is an application deployed as a single unit; a microservice architecture decomposes an application into independently deployable services organized around bounded capabilities.",
        "simple": "A monolith is one deployable application; microservices split it into separately operated services.",
        "example": "A small product may begin as one web application; a high-scale organization may separate billing when independent scaling and ownership justify it.",
        "why": "Architecture affects deployment, team boundaries, failure modes, operational cost, and the speed of coordinated changes.",
        "when": "Start with a modular monolith for a small domain or team; split services when independent ownership, scaling, or deployment benefits outweigh distributed-system costs.",
        "compare": {"other": "microservices", "a": "A monolith packages application capabilities in one deployment unit.", "b": "Microservices deploy capabilities as separate networked processes.", "difference": "A monolith has simpler in-process calls and deployment; microservices introduce network boundaries and independent release cycles.", "pros_a": "Simple local development, transactions, testing, and operations.", "cons_a": "Large teams or components may become tightly coordinated or scale together.", "pros_b": "Independent deployment, scaling, and team ownership.", "cons_b": "Network failures, observability, data consistency, and operational overhead.", "use_a": "Early products, cohesive domains, and small teams.", "use_b": "Stable boundaries with teams and infrastructure able to operate services.", "better": "Prefer the simplest architecture that meets measured needs; microservices are better only when independence benefits exceed their operational cost."},
        "scenario": ("A 4-person team proposes 20 services before launching its first product. What would you recommend?", "Begin with a modular monolith and clear domain boundaries, automate tests and deployment, and gather scaling and ownership evidence. Extract a service only when an independently deployable boundary solves a real problem."),
        "points": ["monolith is one deployment unit", "microservices are separately deployable networked services", "microservices add operational and consistency cost", "choose based on team and measured needs"],
        "misconceptions": [{"trigger": "microservices are always more scalable", "correction": "They enable independent scaling but add network and operations costs and may not improve a small system."}],
    },
    {
        "subject": "Computer Architecture", "topic": "cache memory and main memory", "skills": ["computer architecture", "programming"],
        "book": "A cache is a smaller, faster storage level that retains copies of data likely to be reused, exploiting temporal and spatial locality in a memory hierarchy.",
        "simple": "A cache keeps recently or nearby used data close to the processor so it can be read faster than from main memory.",
        "example": "A CPU cache may keep recently accessed instructions and nearby array elements so repeated loops avoid slower memory accesses.",
        "why": "Caches reduce average access latency by exploiting locality, narrowing the processor-to-memory speed gap.",
        "when": "Use caching when access patterns show reuse and stale data can be controlled; measure hit rate and invalidation costs.",
        "compare": {"other": "main memory (RAM)", "a": "A cache is small, fast, and holds selected copies of data.", "b": "Main memory is larger and holds active programs and data.", "difference": "Caches trade capacity for lower latency and rely on locality; RAM provides larger working capacity at higher latency.", "pros_a": "Fast access for cache hits.", "cons_a": "Limited capacity and complexity managing misses and coherence.", "pros_b": "Much larger working set.", "cons_b": "Higher access latency than cache.", "use_a": "Frequently reused data near a processor or service.", "use_b": "General active data that exceeds cache capacity.", "better": "Use both in a memory hierarchy; neither replaces the other, and cache value depends on locality and hit rate."},
        "scenario": ("A program scans a large array sequentially. Why may this outperform random pointer chasing?", "Sequential access uses spatial locality and often benefits from cache lines and prefetching. Pointer chasing has dependent, scattered accesses that can cause more cache misses."),
        "points": ["cache is smaller and faster", "exploits temporal or spatial locality", "cache misses cost more than hits", "capacity and coherence are trade-offs"],
        "misconceptions": [{"trigger": "cache always contains the newest data", "correction": "Cache coherence or invalidation policies are needed to keep copies consistent; freshness is not automatic in every system."}],
    },
    {
        "subject": "Web Development", "topic": "server-side rendering and client-side rendering", "skills": ["web development", "frontend", "javascript"],
        "book": "Server-side rendering produces HTML on a server for a request; client-side rendering delivers an application that constructs or updates much of the interface in the browser.",
        "simple": "With server rendering, the server sends ready HTML; with client rendering, browser JavaScript builds more of the page.",
        "example": "A news article may be rendered on the server for quick readable content; an interactive dashboard may render much of its UI in the browser.",
        "why": "Rendering strategy affects first content display, interactivity, SEO, server work, caching, and application complexity.",
        "when": "Use server rendering for content where initial HTML, discoverability, or slow-device experience matters; use client rendering for highly interactive applications when its costs are acceptable.",
        "compare": {"other": "client-side rendering", "a": "The server returns HTML for the requested view.", "b": "The browser runs JavaScript to build or update the view.", "difference": "Server rendering can provide meaningful initial markup sooner; client rendering can shift rendering work to the browser after scripts load.", "pros_a": "Readable initial response, discoverability, and good content-first behavior.", "cons_a": "Server rendering cost and hydration or caching complexity.", "pros_b": "Rich interactions and clear separation for app-like clients.", "cons_b": "Large scripts can delay content and require client-side accessibility and SEO care.", "use_a": "Public content pages and performance-sensitive initial views.", "use_b": "Authenticated, highly interactive app surfaces.", "better": "Choose per route and user need; hybrid rendering is often better than forcing one strategy everywhere."},
        "scenario": ("A public product page must appear quickly on mobile and be indexed by search engines. What rendering approach would you evaluate?", "Evaluate server-side or static rendering for the initial product content, then hydrate only needed interactions. Measure real-user performance, ensure semantic HTML, and consider cache freshness and personalization."),
        "points": ["server rendering returns HTML from server", "client rendering builds view in browser", "trade-offs include initial content and interactivity", "select by route and measured user needs"],
        "code": {"language": "html", "question": "Write a semantic HTML button that submits a form.", "answer": "<form action=\"/search\" method=\"get\">\n  <label for=\"query\">Search</label>\n  <input id=\"query\" name=\"q\" type=\"search\">\n  <button type=\"submit\">Search</button>\n</form>", "output": "Submitting the form sends the q field to /search.", "time": "Not applicable to static markup", "space": "Not applicable to static markup"},
    },
    {
        "subject": "Artificial Intelligence", "topic": "search-based AI and machine learning", "skills": ["ai", "programming"],
        "book": "Artificial intelligence studies computational systems that perform tasks associated with intelligent behavior; machine learning is a subfield that fits models or policies from data or interaction.",
        "simple": "AI is the broad goal of making systems act intelligently; machine learning is one way to build AI by learning patterns from examples.",
        "example": "A rule-based path planner can be AI without training; a learned classifier is AI built using machine learning.",
        "why": "Distinguishing explicit search/rules from learned behavior helps choose techniques, explain data needs, and diagnose failures.",
        "when": "Use search or rules when the world and constraints are explicit; use machine learning when patterns are difficult to hand-code and representative data and evaluation are available.",
        "compare": {"other": "machine learning", "a": "Search-based AI explores states using explicit transition and goal definitions.", "b": "Machine learning fits behavior or predictions from examples or feedback.", "difference": "Search relies on a model of actions and goals; learning estimates patterns from data.", "pros_a": "Transparent constraints and guarantees may be available.", "cons_a": "Requires a useful explicit model and can have large search spaces.", "pros_b": "Can capture patterns difficult to specify manually.", "cons_b": "Needs representative data and careful evaluation; behavior may be less interpretable.", "use_a": "Planning with known rules and state transitions.", "use_b": "Prediction or perception from examples.", "better": "Choose based on available knowledge: explicit models favor search; examples with learnable patterns favor ML; hybrid systems are common."},
        "scenario": ("A board game has known legal moves and a finite objective. Would you start with a neural network?", "Not necessarily. Start with search and a well-defined state/action model when feasible, perhaps with heuristics. Use learning if search is too expensive or learned evaluation is justified by data and measured performance."),
        "points": ["AI is the broader field", "ML learns behavior or patterns from data", "search uses explicit states and transitions", "technique follows problem knowledge and evidence"],
    },
    {
        "subject": "Machine Learning", "topic": "supervised and unsupervised learning", "skills": ["machine learning", "data science", "python"],
        "book": "Supervised learning estimates a mapping from inputs to labeled targets; unsupervised learning identifies structure in data without target labels.",
        "simple": "Supervised learning learns from examples with answers; unsupervised learning looks for patterns without supplied answers.",
        "example": "Spam classification uses labeled spam/ham examples; clustering groups customers when no group label is provided.",
        "why": "The task and available labels determine the learning setup, evaluation method, and meaning of model output.",
        "when": "Use supervised learning when reliable target labels exist; use unsupervised learning for structure discovery or representation when targets are unavailable, validating usefulness with domain criteria.",
        "compare": {"other": "unsupervised learning", "a": "Supervised algorithms train against labeled input-target pairs.", "b": "Unsupervised algorithms find structure without target labels.", "difference": "Supervised learning can optimize a known target; unsupervised results require interpretation or downstream validation.", "pros_a": "Directly aligned with a measurable prediction objective.", "cons_a": "Labels cost time and may be biased or noisy.", "pros_b": "Can use unlabeled data to explore structure.", "cons_b": "Clusters or representations may not correspond to useful concepts.", "use_a": "Classification, regression, and forecasting with targets.", "use_b": "Clustering, anomaly exploration, or dimensionality reduction.", "better": "Neither is generally better: use labeled supervision for a known target, unsupervised methods for validated structure discovery."},
        "scenario": ("A bank has historical transactions marked fraudulent or legitimate. What learning setup fits and what leakage risk must be checked?", "Supervised classification fits labeled transactions. Split data by time or entity as appropriate, prevent post-outcome fields from leaking into features, and evaluate precision/recall under class imbalance."),
        "points": ["supervised uses target labels", "unsupervised has no target labels", "examples include classification and clustering", "evaluation must match the task"],
        "misconceptions": [{"trigger": "unsupervised learning has no evaluation", "correction": "It lacks supplied target labels but can still be evaluated using stability, downstream utility, or domain validation."}],
        "code": {"language": "python", "question": "Show the basic supervised-learning workflow using scikit-learn's train_test_split and a classifier.", "answer": "from sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42\n)\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\npredictions = model.predict(X_test)", "output": "predictions contains one predicted label per test example.", "time": "Depends on the estimator, features, and data size", "space": "Depends on the estimator and data size"},
    },
    {
        "subject": "Deep Learning", "topic": "convolutional and transformer neural networks", "skills": ["deep learning", "machine learning"],
        "book": "A convolutional neural network applies learned local filters with shared weights; a transformer uses attention mechanisms to model relationships among sequence elements or other tokens.",
        "simple": "Convolutions detect local patterns with reusable filters; transformers use attention to relate different parts of the input.",
        "example": "A CNN can detect local image edges; a vision transformer divides an image into patches and attends across them.",
        "why": "Architecture choice affects inductive bias, compute, data requirements, and the kind of relationships represented.",
        "when": "Use convolutions when local structure and translation-related inductive bias are useful; use transformers when long-range interactions matter and data/compute support attention.",
        "compare": {"other": "transformer attention", "a": "Convolution applies local kernels with shared parameters.", "b": "Self-attention computes content-dependent interactions among tokens.", "difference": "Convolution is locally structured; standard attention can connect distant positions but its cost often grows quadratically with sequence length.", "pros_a": "Strong local bias and efficient spatial processing.", "cons_a": "Long-range interaction may require depth or larger receptive fields.", "pros_b": "Flexible global relationships and scalable transfer ecosystems.", "cons_b": "Attention can be memory-intensive and data-hungry.", "use_a": "Local image signals or resource-constrained vision tasks.", "use_b": "Sequences or inputs needing global context.", "better": "Benchmark on the task and budget; a CNN may be better for local, efficient vision, while a transformer may help when global context and data justify it."},
        "scenario": ("A small embedded device must classify local defects in images with limited memory. Which architecture family would you investigate first?", "Investigate a compact CNN because local patterns and efficient convolutions may suit the constraint. Compare against compact attention models on the same data and device, measuring latency, memory, and accuracy."),
        "points": ["CNN uses local shared filters", "transformer uses attention across tokens", "attention cost can grow with sequence length", "choose based on data, compute, and task"],
    },
    {
        "subject": "NLP", "topic": "stemming and lemmatization", "skills": ["nlp", "machine learning"],
        "book": "Stemming heuristically removes affixes to produce a stem; lemmatization maps an inflected word to a dictionary lemma using linguistic and often contextual information.",
        "simple": "Stemming chops word endings by rules; lemmatization tries to return a valid base word.",
        "example": "A stemmer may reduce “studies” to “studi”; a lemmatizer can return “study.”",
        "why": "Normalization can reduce vocabulary variation, but aggressive normalization may erase distinctions useful to a task.",
        "when": "Use stemming for lightweight retrieval where rough matching is acceptable; use lemmatization when interpretable valid base forms and linguistic accuracy matter.",
        "compare": {"other": "lemmatization", "a": "Stemming applies heuristic affix rules.", "b": "Lemmatization uses lexical or morphological analysis to find a lemma.", "difference": "Stems may not be real words; lemmas are intended to be valid dictionary forms and may depend on part of speech.", "pros_a": "Fast and language-resource-light.", "cons_a": "Can over-truncate or merge unrelated forms.", "pros_b": "More linguistically meaningful normalization.", "cons_b": "Needs language resources and can cost more.", "use_a": "Large search indexes where rough matching is sufficient.", "use_b": "NLP tasks needing clean lexical forms.", "better": "Use the simplest method whose normalization quality meets task evaluation; lemmatization is often better for linguistic interpretation."},
        "scenario": ("A search engine misses documents because users search for “run” while documents contain “running.” What normalization would you test?", "Test stemming and lemmatization on representative queries and documents, measure retrieval relevance, and check that normalization does not merge meanings that should remain distinct."),
        "points": ["stemming removes affixes heuristically", "lemmatization returns a lexical base form", "stemming can produce nonwords", "evaluate effect on downstream task"],
    },
    {
        "subject": "Generative AI", "topic": "retrieval-augmented generation and fine-tuning", "skills": ["generative ai", "llm", "nlp"],
        "book": "Retrieval-augmented generation conditions a model response on retrieved external information; fine-tuning updates model parameters using task-specific training examples.",
        "simple": "RAG looks up useful information at answer time; fine-tuning teaches the model patterns by changing its weights.",
        "example": "A support assistant can retrieve current policy documents; fine-tuning can make its outputs consistently follow a response format.",
        "why": "The distinction guides whether a need concerns fresh factual context, behavior/style, or specialized task performance.",
        "when": "Use retrieval for changing or citeable knowledge; fine-tune for stable behavior or task patterns when quality data and evaluation justify training.",
        "compare": {"other": "fine-tuning", "a": "RAG retrieves context and supplies it to the model during inference.", "b": "Fine-tuning modifies model parameters using examples.", "difference": "RAG changes available context without changing weights; fine-tuning changes model behavior stored in weights.", "pros_a": "Can use fresh documents and provide source attribution.", "cons_a": "Retrieval quality, context limits, and access control must be managed.", "pros_b": "Can improve consistent format, style, or task behavior.", "cons_b": "Needs curated data and training; does not reliably keep facts current.", "use_a": "Frequently changing, private, or source-grounded knowledge.", "use_b": "Stable task behavior, terminology, or output conventions.", "better": "Use RAG for changing facts, fine-tuning for stable behavior, and combine them only when each solves a measured need."},
        "scenario": ("A company policy changes weekly and answers must cite the current section. Would you fine-tune the model every week?", "Prefer retrieval over the versioned policy corpus, return source references, and evaluate retrieval and grounding. Fine-tune only if stable response behavior needs improvement; it is not a substitute for current source data."),
        "points": ["RAG supplies retrieved context at inference", "fine-tuning updates model parameters", "RAG suits changing source-grounded facts", "fine-tuning suits stable behavior patterns"],
    },
    {
        "subject": "LLM", "topic": "tokens, context windows, and attention", "skills": ["llm", "generative ai", "nlp"],
        "book": "A large language model is a neural sequence model trained at scale to estimate or generate token sequences; its context window bounds the input and output tokens considered in a forward interaction.",
        "simple": "An LLM predicts text in token pieces, and its context window limits how much text it can consider at once.",
        "example": "A long document may exceed a model's context window, so a system can retrieve or summarize relevant sections before asking a question.",
        "why": "Tokenization and context limits affect cost, truncation, retrieval design, and what information is actually available to a response.",
        "when": "Use chunking or retrieval when relevant source material exceeds the context window; verify that important evidence is included rather than assuming the model remembers it.",
        "compare": {"other": "retrieval over external documents", "a": "The context window contains tokens supplied directly for one model interaction.", "b": "Retrieval selects external text and inserts relevant portions into that context.", "difference": "A larger window accepts more direct text but does not itself select relevant evidence; retrieval searches a corpus before inference.", "pros_a": "Simple access to a bounded prompt and its local context.", "cons_a": "Finite token budget, distraction, and cost.", "pros_b": "Scales access to larger corpora and can target relevant sources.", "cons_b": "Depends on indexing, retrieval quality, and source permissions.", "use_a": "Short conversations or focused documents.", "use_b": "Large or changing knowledge bases.", "better": "Use direct context for small inputs; use retrieval when the corpus exceeds practical context limits or requires source grounding."},
        "scenario": ("A user asks about a 500-page manual, but the model accepts only a bounded context. What system design would you use?", "Index the manual into permission-aware chunks with metadata, retrieve relevant passages for each query, fit them into the context budget, cite the passages, and test retrieval recall and answer grounding."),
        "points": ["LLMs process token sequences", "context window is finite", "larger context does not ensure relevance", "retrieval can select external evidence"],
    },
    {
        "subject": "Computer Vision", "topic": "image classification and object detection", "skills": ["computer vision", "deep learning"],
        "book": "Image classification assigns one or more labels to an image or region; object detection predicts object classes and spatial bounding boxes within an image.",
        "simple": "Classification says what is in an image; detection says what objects are present and where they are.",
        "example": "A classifier labels an image “dog”; a detector labels a dog and returns a box around it.",
        "why": "Task definition determines annotation requirements, model output, and appropriate evaluation metrics.",
        "when": "Use classification when an image-level label is sufficient; use detection when object locations and multiple instances matter.",
        "compare": {"other": "object detection", "a": "Classification predicts image- or crop-level classes.", "b": "Detection predicts classes and bounding boxes for instances.", "difference": "Detection localizes instances and generally requires box annotations; classification does not inherently provide locations.", "pros_a": "Simpler labels and output for image-level decisions.", "cons_a": "Cannot report where objects appear or count instances reliably.", "pros_b": "Provides instance location and class.", "cons_b": "More complex labels, models, and evaluation.", "use_a": "Whole-image categories or quality labels.", "use_b": "Surveillance, inventory, or robotics needing object positions.", "better": "Choose classification for image-level decisions and detection when localization changes the downstream action."},
        "scenario": ("A warehouse robot must pick one item among several objects in a camera frame. Which vision task is needed?", "Object detection is a suitable starting task because the robot needs object classes and locations. If precise shape or grasp boundaries are required, evaluate instance segmentation or pose estimation instead."),
        "points": ["classification predicts image-level label", "detection predicts class and location", "detection needs localization annotations", "task follows downstream requirement"],
    },
    {
        "subject": "Cybersecurity", "topic": "authentication and authorization", "skills": ["security", "programming"],
        "book": "Authentication verifies the identity or credentials of a principal; authorization determines which actions that authenticated principal is permitted to perform.",
        "simple": "Authentication asks “who are you?”; authorization asks “what are you allowed to do?”",
        "example": "A user signs in with a passkey (authentication), then policy permits that user to view but not edit a report (authorization).",
        "why": "Separating identity proof from permission checks prevents a successful login from granting inappropriate access.",
        "when": "Authenticate at trust boundaries and authorize every protected operation using least privilege and current resource context.",
        "compare": {"other": "authorization", "a": "Authentication verifies a principal's identity.", "b": "Authorization evaluates the principal's allowed actions.", "difference": "Identity proof does not imply permission; authorization uses identity, roles, policies, and resource context.", "pros_a": "Establishes an accountable identity or credential.", "cons_a": "Strong login alone does not prevent excessive access.", "pros_b": "Enforces least-privilege access to resources.", "cons_b": "Incorrect or stale policies can block valid work or expose data.", "use_a": "Login, service identity, and credential verification.", "use_b": "Every operation that accesses a protected resource.", "better": "Both are required in most protected systems; neither substitutes for the other."},
        "scenario": ("A user can log in successfully but can change another tenant's invoice by altering an ID in the URL. What failed?", "Authorization and object-level access control failed. Check the authenticated principal's tenant and permission against the requested invoice on every request; do not trust a client-supplied identifier."),
        "points": ["authentication verifies identity", "authorization grants or denies actions", "login does not imply access to every resource", "enforce least privilege per resource"],
        "misconceptions": [{"trigger": "authenticated users are authorized for all data", "correction": "Authentication proves identity; permissions still need to be checked for each protected action and resource."}],
    },
    {
        "subject": "Cloud Computing", "topic": "containers and virtual machines", "skills": ["cloud", "devops"],
        "book": "A virtual machine virtualizes hardware to run a guest operating system; a container isolates processes while sharing the host operating-system kernel.",
        "simple": "A VM includes a guest OS; a container packages an application environment but shares the host kernel.",
        "example": "A VM can run a different guest kernel; containers can start lightweight copies of a web service on a compatible host.",
        "why": "Isolation, startup time, density, kernel requirements, and operational boundaries differ between the two.",
        "when": "Use containers for portable, dense application packaging on compatible kernels; use VMs when stronger OS isolation or a different guest OS/kernel is required.",
        "compare": {"other": "virtual machines", "a": "Containers isolate processes while sharing a host kernel.", "b": "VMs emulate or virtualize hardware and run guest operating systems.", "difference": "Containers typically share the kernel; VMs include separate guest OS instances.", "pros_a": "Fast startup and efficient resource use for compatible workloads.", "cons_a": "Kernel sharing and host configuration affect isolation and compatibility.", "pros_b": "Strong OS boundary and support for different guest kernels.", "cons_b": "More resource use and slower startup in many configurations.", "use_a": "Microservices and repeatable application deployment.", "use_b": "Legacy OS needs, kernel differences, or stronger isolation requirements.", "better": "Containers often suit application packaging; VMs are better when OS-level separation or kernel flexibility is required."},
        "scenario": ("A team must run a legacy workload requiring a different kernel from its cloud host. Which isolation option fits?", "A VM is generally appropriate because it can run a guest operating system with its own kernel. Confirm the cloud platform supports the guest and weigh resource overhead and security configuration."),
        "points": ["containers share host kernel", "VMs run guest operating systems", "containers are generally lighter", "VMs support kernel separation"],
    },
    {
        "subject": "DevOps", "topic": "continuous integration and continuous delivery/deployment", "skills": ["devops", "git", "testing"],
        "book": "Continuous integration frequently integrates changes and validates them automatically; continuous delivery keeps changes releasable, while continuous deployment automatically releases qualifying changes to production.",
        "simple": "CI checks changes often; continuous delivery keeps a release ready; continuous deployment releases validated changes automatically.",
        "example": "A pull request runs tests in CI; a delivery pipeline stages a release for approval; a deployment pipeline can automatically promote it after checks.",
        "why": "Automation shortens feedback loops and makes release risk visible, but does not replace reliable tests, observability, or rollback plans.",
        "when": "Adopt CI for shared code and fast validation; choose manual release approval or automatic production deployment based on risk, regulation, and operational maturity.",
        "compare": {"other": "continuous deployment", "a": "Continuous delivery prepares validated changes for release, commonly with a human decision.", "b": "Continuous deployment automatically releases qualifying changes to production.", "difference": "Both automate validation; deployment removes the explicit release approval step.", "pros_a": "Release-ready software with a controlled launch decision.", "cons_a": "A manual gate can delay releases.", "pros_b": "Fast, frequent releases with little manual intervention.", "cons_b": "Requires strong tests, monitoring, and recovery; mistakes can reach users quickly.", "use_a": "Regulated, high-risk, or approval-controlled environments.", "use_b": "Mature systems with confidence in automated controls.", "better": "Choose automatic deployment only when risk controls and recovery are strong; otherwise continuous delivery provides a useful approval gate."},
        "scenario": ("A small team wants safe daily releases but production changes need product-owner approval. What pipeline would you implement?", "Use CI on every change and continuous delivery that deploys to a verified staging environment, then requires the product-owner approval before production. Automate smoke tests, monitoring, and rollback."),
        "points": ["CI integrates and validates changes", "delivery keeps a release-ready artifact", "deployment automatically releases to production", "automation requires tests and recovery controls"],
    },
    {
        "subject": "System Design", "topic": "vertical and horizontal scaling", "skills": ["system design", "cloud", "backend"],
        "book": "Vertical scaling increases the resources of a single computing node; horizontal scaling adds nodes and distributes workload among them.",
        "simple": "Vertical scaling makes one machine bigger; horizontal scaling adds more machines.",
        "example": "Increasing a database server's RAM is vertical scaling; adding application instances behind a load balancer is horizontal scaling.",
        "why": "Scaling strategy affects capacity limits, availability, coordination, cost, and application complexity.",
        "when": "Scale vertically for simplicity while a node has headroom; scale horizontally when capacity, availability, or independent workload growth requires multiple nodes.",
        "compare": {"other": "horizontal scaling", "a": "Vertical scaling increases CPU, memory, or storage on one node.", "b": "Horizontal scaling adds nodes and distributes work.", "difference": "Vertical scaling is simpler but bounded by one machine; horizontal scaling can add capacity but introduces distribution and coordination.", "pros_a": "Simple application and data model.", "cons_a": "Hardware ceiling and single-node failure exposure.", "pros_b": "Potential capacity and availability through multiple nodes.", "cons_b": "Load balancing, shared state, consistency, and operations become harder.", "use_a": "Early growth, stateful workloads, or workloads not yet partitioned.", "use_b": "Stateless services or workloads with partitionable data and demand.", "better": "Use vertical scaling for simplicity until its limits; horizontal scaling is better when workload and design support safe distribution."},
        "scenario": ("A stateless API is CPU-bound and traffic doubles during predictable campaigns. How would you scale it?", "Add or autoscale stateless instances behind a load balancer, define capacity and health checks, and test downstream database limits. Vertical upgrades may be a short-term step but do not remove a single-node ceiling."),
        "points": ["vertical adds resources to one node", "horizontal adds nodes", "horizontal introduces distribution complexity", "scale according to bottleneck and availability needs"],
    },
    {
        "subject": "Distributed Systems", "topic": "strong consistency and eventual consistency", "skills": ["distributed systems", "backend", "cloud"],
        "book": "Strong consistency requires reads to reflect completed writes according to a defined ordering model; eventual consistency permits temporary replica divergence but converges when updates cease and propagation succeeds.",
        "simple": "Strong consistency makes users see agreed updates immediately under its model; eventual consistency allows short-lived differences that later converge.",
        "example": "A bank balance update may require strong coordination; a social feed's view count may tolerate delayed replica updates.",
        "why": "Consistency choices trade coordination latency and availability behavior against how quickly all readers observe updates.",
        "when": "Use stronger guarantees for invariants such as financial transfers; allow eventual consistency for data where bounded staleness is acceptable and reconciliation is defined.",
        "compare": {"other": "eventual consistency", "a": "Strong consistency provides a specified immediate ordering of completed writes.", "b": "Eventual consistency permits temporary divergence with convergence under stated assumptions.", "difference": "Strong guarantees may require coordination; eventual systems can respond amid propagation delay but expose stale reads.", "pros_a": "Simpler reasoning about current state and invariants.", "cons_a": "Coordination can increase latency or reduce availability during partitions.", "pros_b": "Can improve responsiveness and availability for some replicated workloads.", "cons_b": "Clients must tolerate stale reads, conflicts, or reconciliation.", "use_a": "Balances, inventory reservations, and strict uniqueness.", "use_b": "Feeds, analytics counters, or replicated preferences with tolerated delay.", "better": "Use the weakest consistency that preserves business invariants; stronger is not automatically better for every user-facing feature."},
        "scenario": ("A shopping cart count may lag, but inventory must not be oversold. Where can eventual consistency be tolerated?", "The displayed cart count can often tolerate bounded eventual consistency. Inventory reservation and final purchase need an atomic or otherwise invariant-preserving mechanism, such as conditional updates or reservations."),
        "points": ["strong consistency defines immediate ordering guarantees", "eventual consistency allows temporary staleness", "strong coordination can cost latency or availability", "business invariants determine acceptable consistency"],
    },
    {
        "subject": "Blockchain", "topic": "proof of work and proof of stake", "skills": ["blockchain", "security", "distributed systems"],
        "book": "Proof of work selects block producers through computational work; proof of stake selects or weights validators based on economically committed stake and protocol rules.",
        "simple": "Proof of work uses computing effort to secure block proposals; proof of stake uses locked economic stake and penalties or rewards.",
        "example": "A proof-of-work miner expends hash computation; a proof-of-stake validator risks a deposit if protocol rules are violated.",
        "why": "Consensus mechanisms determine Sybil resistance, resource use, incentives, finality behavior, and attack assumptions.",
        "when": "Evaluate a consensus protocol against threat model, decentralization goals, energy and hardware constraints, validator economics, and recovery procedures.",
        "compare": {"other": "proof of stake", "a": "Proof of work makes participation costly through computational energy and hardware.", "b": "Proof of stake makes participation depend on committed stake and protocol penalties.", "difference": "Their resource costs and attack incentives differ; neither removes the need for honest-majority or protocol assumptions.", "pros_a": "A long-studied cost mechanism with open participation subject to hardware economics.", "cons_a": "High energy and specialized hardware concentration risks.", "pros_b": "Lower direct energy use and economic penalties can deter misconduct.", "cons_b": "Stake distribution, governance, and protocol-specific attack risks matter.", "use_a": "Networks whose security model accepts work-based resource expenditure.", "use_b": "Networks able to define robust staking, slashing, and validator rules.", "better": "No universal winner; choose only after comparing protocol assumptions, decentralization, energy, incentives, and attack resistance."},
        "scenario": ("A consortium considers a chain for internal audit records where participants are known organizations. Is public proof of work necessary?", "Not automatically. First test whether a blockchain is needed at all; a conventional append-only signed log may suffice. If shared consensus is required, a permissioned protocol can match known participants and governance better than public proof of work."),
        "points": ["proof of work uses computational work", "proof of stake uses economic stake and protocol rules", "each has distinct resource and attack assumptions", "choose from threat model and governance"],
    },
    {
        "subject": "Data Science", "topic": "exploratory analysis and confirmatory analysis", "skills": ["data science", "statistics", "sql"],
        "book": "Exploratory data analysis uses summaries and visualizations to reveal patterns and assess data quality; confirmatory analysis evaluates prespecified hypotheses using an appropriate statistical design.",
        "simple": "Exploratory analysis looks for patterns; confirmatory analysis tests a specific claim using a planned method.",
        "example": "Plotting weekly sales to notice seasonality is exploratory; testing a prespecified campaign effect with a control group is confirmatory.",
        "why": "Separating exploration from confirmation reduces the risk of treating patterns discovered in the same data as independent proof.",
        "when": "Explore to form questions and check data; confirm on held-out data or a prespecified experiment with suitable assumptions and uncertainty estimates.",
        "compare": {"other": "confirmatory analysis", "a": "Exploratory analysis searches data for patterns and generates hypotheses.", "b": "Confirmatory analysis tests planned hypotheses using specified methods.", "difference": "Exploration is flexible and discovery-oriented; confirmation is constrained to control false-positive risk and quantify uncertainty.", "pros_a": "Finds data issues and useful questions.", "cons_a": "Repeated searching can produce chance patterns.", "pros_b": "Provides a disciplined test of a defined claim.", "cons_b": "Can miss unexpected patterns and depends on valid assumptions.", "use_a": "Initial understanding, quality checks, and hypothesis generation.", "use_b": "Decision evidence after defining the question and analysis plan.", "better": "Use both sequentially: explore, then confirm with independent evidence or a prespecified design."},
        "scenario": ("An analyst tries 40 segment definitions and reports the one with the largest uplift as statistically significant. What concern arises?", "Multiple comparisons and selection bias make the reported significance optimistic. Treat the result as exploratory, adjust for the search or validate the selected hypothesis on independent data or a new experiment."),
        "points": ["exploration discovers patterns", "confirmation tests prespecified hypotheses", "multiple searches inflate false positives", "validate findings independently"],
    },
    {
        "subject": "Software Testing", "topic": "unit and integration testing", "skills": ["testing", "software engineering", "qa"],
        "book": "Unit testing verifies a small component in isolation; integration testing verifies interactions among components or systems at their interfaces.",
        "simple": "A unit test checks one small piece; an integration test checks that connected pieces work together.",
        "example": "Test a tax calculation function alone as a unit; test checkout with a real test database and payment adapter as integration.",
        "why": "Different test levels reveal different defects and trade off execution speed, isolation, and confidence in real collaboration.",
        "when": "Use many focused unit tests for logic and integration tests around important boundaries, supplemented by end-to-end checks for critical user journeys.",
        "compare": {"other": "integration testing", "a": "Unit tests isolate a small component, often replacing collaborators.", "b": "Integration tests exercise real interactions across component boundaries.", "difference": "Unit failures localize logic defects quickly; integration tests expose wiring, protocol, and data issues.", "pros_a": "Fast, focused, and easy to diagnose.", "cons_a": "Mocks can hide integration defects.", "pros_b": "Validates contracts and real dependencies.", "cons_b": "Slower and may require environment setup.", "use_a": "Algorithms, validation, and business rules.", "use_b": "Database, API, and service boundaries.", "better": "Neither replaces the other; prioritize fast unit feedback and targeted integration tests for high-risk boundaries."},
        "scenario": ("All service unit tests pass, but production requests fail when saving to the database. Which tests are missing?", "Add integration tests that exercise the service-to-database mapping, schema, transaction, and error behavior against a test database or faithful containerized dependency."),
        "points": ["unit tests isolate components", "integration tests check component interactions", "mocks can miss boundary defects", "use a balanced test strategy"],
        "code": {"language": "python", "question": "Write a unit test for a function that adds two integers using unittest.", "answer": "import unittest\n\n\ndef add(left, right):\n    return left + right\n\n\nclass AddTests(unittest.TestCase):\n    def test_adds_two_integers(self):\n        self.assertEqual(add(2, 3), 5)", "output": "The test passes when add(2, 3) returns 5.", "time": "O(1) for the example operation", "space": "O(1)"},
    },
    {
        "subject": "Git and GitHub", "topic": "merge and rebase", "skills": ["git", "devops"],
        "book": "A merge combines histories by creating a commit with multiple parents when needed; rebase reapplies commits onto a new base, producing a rewritten linear history.",
        "simple": "Merge joins branch histories; rebase moves your commits onto a newer base and changes their commit IDs.",
        "example": "Merge a shared feature branch to preserve its history; rebase a private local branch to update it before review.",
        "why": "The choice affects history shape, collaboration, conflict resolution, and whether published commit identities change.",
        "when": "Merge shared or published work safely; rebase private, unpublished commits when a linear history helps, and coordinate before rewriting shared history.",
        "compare": {"other": "rebase", "a": "Merge records the join of two histories without rewriting existing commits.", "b": "Rebase reapplies commits onto another base, creating new commit identities.", "difference": "Merge preserves topology; rebase can linearize but rewrites the rebased commits.", "pros_a": "Safe for shared history and shows branch integration.", "cons_a": "Can produce a more branched history.", "pros_b": "Linear history can be easier to read or bisect.", "cons_b": "Rewriting published commits can disrupt collaborators.", "use_a": "Integrating shared or already-published branches.", "use_b": "Cleaning a private branch before sharing.", "better": "Use merge for shared history; rebase private work only when its history rewrite is safe and understood."},
        "scenario": ("A teammate has pushed a branch others are using. You need to bring main into it. Should you force-rebase it?", "Avoid rewriting the shared branch without coordination. Merge main into the branch or agree with all users on a coordinated rebase and safe force-push procedure."),
        "points": ["merge preserves existing commit identities", "rebase rewrites commits", "rebase can create linear history", "avoid rewriting shared history unexpectedly"],
    },
    {
        "subject": "API and Web Services", "topic": "REST and RPC", "skills": ["api", "backend", "web development"],
        "book": "REST is an architectural style using resource-oriented representations and constraints such as stateless interactions; RPC models a request as invoking a named operation on a remote service.",
        "simple": "REST focuses on resources and standard HTTP semantics; RPC focuses on calling a specific remote operation.",
        "example": "REST might use PATCH /users/42; RPC might call UpdateUser(id=42).",
        "why": "The interaction model influences API discoverability, client coupling, protocol tooling, and how operations map to domain behavior.",
        "when": "Use resource-oriented REST for broadly interoperable CRUD-like resources; use RPC for explicit operations, internal typed contracts, or streaming where tooling supports it.",
        "compare": {"other": "RPC", "a": "REST organizes interactions around resources and representations.", "b": "RPC organizes interactions around named procedures or operations.", "difference": "REST uses resource semantics and HTTP methods; RPC makes operation names and request schemas explicit.", "pros_a": "Widely understood HTTP conventions and cache-friendly resource patterns.", "cons_a": "Complex actions can fit awkwardly into resource endpoints.", "pros_b": "Clear operation semantics and strong generated-client support in many frameworks.", "cons_b": "Can couple clients closely to service method contracts.", "use_a": "Public web APIs and resource lifecycle operations.", "use_b": "Internal service calls or action-oriented and streaming interfaces.", "better": "Choose based on clients, operations, caching, and tooling; neither style is universally superior."},
        "scenario": ("An internal service needs strongly typed low-latency calls between teams using one controlled stack. Which API style might fit?", "RPC with a versioned typed contract may fit, especially if tooling supports compatibility and observability. Confirm latency and client coupling needs; REST may be preferable for broader interoperability."),
        "points": ["REST is resource-oriented", "RPC is operation-oriented", "REST uses HTTP semantics and representations", "choose by interoperability and contract needs"],
        "code": {"language": "python", "question": "Write a small REST-style endpoint signature for retrieving a user by ID using FastAPI.", "answer": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get(\"/users/{user_id}\")\ndef get_user(user_id: int):\n    return {\"id\": user_id}", "output": "GET /users/42 returns a JSON user resource.", "time": "Depends on the backing storage", "space": "Depends on the response"},
    },
    {
        "subject": "IoT", "topic": "edge computing and cloud computing", "skills": ["iot", "cloud", "networking"],
        "book": "Edge computing processes data near its source or point of use; cloud computing provides on-demand shared computing resources through remote infrastructure.",
        "simple": "Edge computing works near the device; cloud computing works in remote data centers.",
        "example": "A factory camera can reject defective items locally and send summarized metrics to a cloud analytics service.",
        "why": "Placement determines response latency, network dependence, privacy exposure, and available compute capacity.",
        "when": "Use edge processing for low-latency, disconnected, or data-minimizing decisions; use cloud processing for elastic, centralized, or compute-intensive workloads.",
        "compare": {"other": "cloud computing", "a": "Edge computing runs processing near devices or data sources.", "b": "Cloud computing runs processing in remote shared infrastructure.", "difference": "Edge reduces dependence on a round trip; cloud centralizes elastic capacity and management.", "pros_a": "Low latency, offline operation, and reduced raw-data transfer.", "cons_a": "Limited resources and harder fleet updates.", "pros_b": "Elastic compute, central management, and broad managed services.", "cons_b": "Network latency, connectivity dependence, and data-transfer concerns.", "use_a": "Real-time control, privacy-sensitive filtering, and intermittent networks.", "use_b": "Large analytics jobs, central aggregation, and elastic workloads.", "better": "A hybrid is often appropriate: decide locally when necessary and aggregate or train centrally when useful."},
        "scenario": ("A sensor must stop a dangerous machine within milliseconds even if internet connectivity is lost. Where should the safety decision run?", "Run the safety-critical control at the edge or in a local controller with a defined fail-safe state. The cloud may monitor and analyze data, but should not be on the time-critical safety path."),
        "points": ["edge runs near data source", "cloud offers remote elastic resources", "edge helps latency and offline behavior", "cloud supports centralized heavy workloads"],
    },
    {
        "subject": "Theory of Computation", "topic": "P, NP, and NP-completeness", "skills": ["theory of computation", "algorithms"],
        "book": "P is the class of decision problems solvable in polynomial time by a deterministic machine; NP is the class whose proposed solutions can be verified in polynomial time; NP-complete problems are in NP and NP-hard under polynomial-time reductions.",
        "simple": "P problems can be solved quickly in polynomial time; NP problems have answers that can be checked quickly; NP-complete problems are among the hardest in NP.",
        "example": "Given a proposed Hamiltonian cycle, checking that every vertex is visited once is polynomial; finding one is an NP-complete decision problem.",
        "why": "These complexity classes help reason about algorithm limits and whether exact solutions are likely to scale.",
        "when": "Use reductions and class membership to assess exact algorithms; for hard optimization instances, investigate approximation, heuristics, parameterized methods, or restricted inputs.",
        "compare": {"other": "NP-complete problems", "a": "P contains decision problems solvable in polynomial time.", "b": "NP-complete problems are in NP and every NP problem reduces to them in polynomial time.", "difference": "P describes efficient solvability; NP-completeness identifies problems at least as hard as all NP problems under reductions.", "pros_a": "Polynomial algorithms provide scalable exact solutions in the input-size model.", "cons_a": "Not every practical problem is known to be in P.", "pros_b": "A proof communicates broad hardness evidence and guides alternatives.", "cons_b": "It does not prove no practical or approximate solution exists.", "use_a": "Classifying known tractable decision problems.", "use_b": "Establishing hardness for exact general-case problems.", "better": "These are not competing techniques; use complexity classification to choose exact, approximate, or restricted approaches."},
        "scenario": ("A new scheduling problem is suspected to be NP-hard. How would you support that claim and still build a useful product?", "Give a polynomial-time reduction from a known NP-hard problem for the hardness claim. Then examine input limits, special cases, approximation guarantees, heuristics, and acceptable optimality gaps for product needs."),
        "points": ["P problems are polynomial-time solvable", "NP solutions are polynomial-time verifiable", "NP-complete means in NP and NP-hard", "hardness does not rule out useful approximations"],
    },
    {
        "subject": "Compiler Design", "topic": "compiler and interpreter", "skills": ["compiler design", "programming"],
        "book": "A compiler translates a source program into another representation, often machine code, before execution; an interpreter executes a program by processing its representation at runtime.",
        "simple": "A compiler translates a program before running it; an interpreter executes program instructions as it processes them.",
        "example": "A C compiler can produce a native executable; a bytecode virtual machine interprets or just-in-time compiles bytecode.",
        "why": "Execution strategy affects startup, optimization, portability, diagnostics, and deployment requirements.",
        "when": "Use ahead-of-time compilation for standalone optimized binaries; use interpretation or virtual machines for portability, dynamic behavior, or interactive workflows.",
        "compare": {"other": "interpreter", "a": "A compiler translates source to a target representation before execution.", "b": "An interpreter processes or executes program representation at runtime.", "difference": "Compilation can move work earlier and optimize globally; interpretation may enable flexibility and immediate execution.", "pros_a": "Can produce efficient standalone code and catch some errors before running.", "cons_a": "Compilation and platform-specific artifacts may be needed.", "pros_b": "Interactive execution and flexible runtime behavior.", "cons_b": "Runtime interpretation can add overhead and deployment dependencies.", "use_a": "Production binaries and workloads benefiting from ahead-of-time optimization.", "use_b": "Scripting, REPLs, and portable virtual-machine runtimes.", "better": "Many systems combine both, such as bytecode plus JIT; choose for deployment, optimization, and language requirements."},
        "scenario": ("A language needs interactive development and high performance for frequently executed code. Must it choose only compilation or interpretation?", "No. It can interpret initially and use just-in-time compilation for hot paths. Measure startup, optimization overhead, peak performance, and runtime complexity."),
        "points": ["compiler translates representation before execution", "interpreter processes at runtime", "compilation can enable ahead-of-time optimization", "hybrid systems can combine techniques"],
    },
]


def _normal(value: str) -> str:
    return " ".join(value.casefold().split())


def _answer_for(record: dict[str, Any], kind: str) -> str:
    comparison = record["compare"]
    if kind == "definition":
        return f"Definition: {record['book']}\n\nSimple answer: {record['simple']}\n\nExample: {record['example']}"
    if kind == "comparison":
        return (
            f"{record['topic']} vs. {comparison['other']}\n\n"
            f"Option A — {record['topic']}: {comparison['a']}\n"
            f"Advantages: {comparison['pros_a']}\nDisadvantages: {comparison['cons_a']}\n"
            f"Use when: {comparison['use_a']}\n\n"
            f"Option B — {comparison['other']}: {comparison['b']}\n"
            f"Advantages: {comparison['pros_b']}\nDisadvantages: {comparison['cons_b']}\n"
            f"Use when: {comparison['use_b']}\n\n"
            f"Difference: {comparison['difference']}\n"
            f"Which is better: {comparison['better']}"
        )
    if kind == "why":
        return f"{record['why']} Example: {record['example']}"
    if kind == "when_to_use":
        return record["when"]
    if kind == "scenario":
        return record["scenario"][1]
    if kind == "coding":
        code = record["code"]
        return (
            f"Approach: Apply the stated operation directly and handle its input contract.\n"
            f"```{code['language']}\n{code['answer']}\n```\n"
            f"Example: {code['output']}\nTime: {code['time']}\nSpace: {code['space']}"
        )
    raise ValueError(f"Unsupported question kind: {kind}")


def _question_for(record: dict[str, Any], kind: str, difficulty: str) -> str:
    topic = record["topic"]
    comparison = record["compare"]
    prompts = {
        "easy": {
            "definition": f"In simple terms, what is {topic}? Give an example.",
            "comparison": f"What is one important difference between {topic} and {comparison['other']}?",
            "why": f"Why is {topic} useful?",
            "when_to_use": f"When would you use {topic}?",
            "scenario": record["scenario"][0],
            "coding": record.get("code", {}).get("question", ""),
        },
        "intermediate": {
            "definition": f"Define {topic}, explain how it works, and give a practical example.",
            "comparison": f"Compare {topic} with {comparison['other']}, including their main differences, advantages, disadvantages, and use cases.",
            "why": f"Why might an engineer choose {topic}, and what trade-off should they consider?",
            "when_to_use": f"When should an engineer use {topic}, and when might an alternative be preferable?",
            "scenario": record["scenario"][0],
            "coding": record.get("code", {}).get("question", ""),
        },
        "advanced": {
            "definition": f"Explain the design principles behind {topic}, its limitations, and where it fits in a real system.",
            "comparison": f"Compare {topic} and {comparison['other']} in depth. Explain both options, differences, advantages, disadvantages, use cases, and which is better under different constraints.",
            "why": f"Why can {topic} be appropriate in one system but harmful in another? Justify the trade-offs.",
            "when_to_use": f"Given a real workload, when would you choose {topic} over alternatives? State assumptions and failure cases.",
            "scenario": record["scenario"][0],
            "coding": record.get("code", {}).get("question", ""),
        },
    }
    return prompts[difficulty][kind]


def build_records() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        title = subject["subject"]
        role_names = ROLE_SUBJECTS[title]
        kinds = ["definition", "comparison", "why", "when_to_use", "scenario"]
        if subject.get("code"):
            kinds.append("coding")
        for difficulty in ("easy", "intermediate", "advanced"):
            for kind in kinds:
                key_points = subject["points"][:]
                if kind == "comparison":
                    key_points = [
                        "explain both options", "state their difference",
                        "give advantages and disadvantages", "identify use cases",
                        "recommend conditionally based on situation",
                    ]
                elif kind == "coding":
                    key_points = ["provide working code", "explain the approach", "give a valid example", "state time and space complexity"]
                elif kind == "scenario":
                    key_points = ["identify the relevant constraint", "choose an appropriate approach", "justify the trade-off"]
                elif kind == "when_to_use":
                    key_points = ["state suitable conditions", "state an alternative or limitation"]
                elif kind == "why":
                    key_points = ["explain the purpose", "describe a benefit", "mention a relevant trade-off"]

                record = {
                    "id": f"{_normal(title).replace(' ', '_')}_{kind}_{difficulty}",
                    "subject": title,
                    "topic": subject["topic"],
                    "skills": subject["skills"],
                    "job_roles": role_names,
                    "difficulty": difficulty,
                    "question_type": kind,
                    "question": _question_for(subject, kind, difficulty),
                    "simple_answer": subject["simple"],
                    "book_definition": subject["book"],
                    "example": subject["example"],
                    "reference_answer": _answer_for(subject, kind),
                    "key_points": key_points,
                    "common_misconceptions": subject.get("misconceptions", []),
                    "comparison": subject["compare"],
                    "scenario": {"question": subject["scenario"][0], "answer": subject["scenario"][1]},
                }
                if subject.get("code"):
                    record["coding_example"] = subject["code"]
                if title == "Algorithms" and subject["topic"] == "binary search" and kind == "definition":
                    record["evaluation_rubric"] = {
                        "core_points": [
                            {
                                "label": "Identifies binary search as a way to find a target",
                                "cues": ["searching algorithm", "search algorithm", "find an element", "find a target"],
                            },
                            {
                                "label": "States that the input must be sorted",
                                "cues": ["sorted array", "sorted data", "sorted sequence", "ordered array"],
                            },
                            {
                                "label": "Explains that the search range is repeatedly halved",
                                "cues": ["halves", "half of the search", "dividing the search range into two", "eliminating half"],
                            },
                        ],
                        "refinement_points": [
                            {
                                "label": "Mentions comparing the target with the middle element",
                                "cues": ["middle element", "middle value", "middle of the", "midpoint"],
                            },
                        ],
                        "suggestion": "comparing the target with the middle element",
                        "contradictions": [
                            {
                                "claim": "Binary search can be used without an ordered search space.",
                                "correction": "Binary search requires sorted data or a valid monotonic predicate.",
                                "cues": [["binary search", "searching algorithm", "search algorithm"]],
                                "negated_term": "sorted",
                            },
                        ],
                        "improved_answer": (
                            "Binary search is a searching algorithm used to find an element in a sorted array "
                            "by comparing the target with the middle element and repeatedly eliminating half "
                            "of the search space."
                        ),
                    }
                output.append(record)
    for subject in EXTRA_TOPICS:
        for difficulty in ("easy", "intermediate", "advanced"):
            question = subject["question"]
            if difficulty == "easy":
                question = subject.get("simple_question", question)
            elif difficulty == "advanced":
                question = f"Give a detailed, situation-based answer: {question}"
            output.append({
                "id": f"{_normal(subject['topic']).replace(' ', '_')}_{difficulty}",
                "subject": subject["subject"],
                "topic": subject["topic"],
                "skills": subject["skills"],
                "job_roles": subject["job_roles"],
                "difficulty": difficulty,
                "question_type": subject["question_type"],
                "question": question,
                "simple_answer": subject["simple_answer"],
                "book_definition": subject["book_definition"],
                "example": subject["example"],
                "reference_answer": subject["reference_answer"],
                "key_points": subject["key_points"],
                "common_misconceptions": subject.get("common_misconceptions", []),
                **({"comparison": subject["comparison"]} if "comparison" in subject else {}),
            })
    return output


def main() -> None:
    records = build_records()
    if len({record["subject"] for record in records}) != 30:
        raise ValueError("The dataset must include exactly the 30 requested subjects.")
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    ROLE_PATH.write_text(
        json.dumps(
            {
                name: {"skills": skills, "subjects": [subject for subject, roles in ROLE_SUBJECTS.items() if name in roles]}
                for name, skills in ROLE_SKILLS.items()
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} questions across 30 subjects and {len(ROLE_SKILLS)} job roles.")


if __name__ == "__main__":
    main()
