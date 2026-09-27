"""Generate a deterministic educational CS interview corpus from curated topic notes.

This seed generator creates a broad starter corpus, not a substitute for a large,
reviewed dataset. Keep validation and test topics separate from training topics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent

# category, topic, explanation, interview question, example, complexity
TOPICS: list[tuple[str, str, str, str, str, dict[str, str]]] = [
    ("dsa", "arrays", "An array stores elements in contiguous indexed positions. Index access is constant time, while insertion in the middle may require shifting later elements.", "When would you choose an array over a linked list?", "Read values[3] directly to access the fourth element.", {"access": "O(1)", "middle insertion": "O(n)"}),
    ("dsa", "strings", "A string is a sequence of characters. Many languages make strings immutable, so repeated concatenation can allocate new strings; a builder or list can be more efficient.", "Why can repeated string concatenation be inefficient?", "Collect fragments in a list and join once instead of repeatedly copying a growing string.", {"index": "O(1)", "search": "O(n)"}),
    ("dsa", "linked lists", "A linked list stores values in nodes connected by references. Insertion or deletion is efficient when the node is already known, but reaching an arbitrary index requires traversal.", "What is one trade-off between an array and a linked list?", "A singly linked node stores a value and a reference to the next node.", {"search": "O(n)", "insert after known node": "O(1)"}),
    ("dsa", "stacks", "A stack follows last-in, first-out order. Push and pop operate at the top; stacks are useful for nested calls, undo history, and delimiter matching.", "How would you use a stack to check balanced brackets?", "Push opening brackets; on a closing bracket, check and remove the matching opener.", {"push": "O(1)", "pop": "O(1)"}),
    ("dsa", "queues", "A queue follows first-in, first-out order. An efficient queue adds at the rear and removes from the front, which is useful for scheduling and breadth-first search.", "Why is a queue appropriate for breadth-first search?", "A FIFO queue visits graph vertices in layers of increasing distance.", {"enqueue": "O(1)", "dequeue": "O(1)"}),
    ("dsa", "hash maps", "A hash map maps keys to values using a hash function. Lookup is expected constant time with a healthy load factor, but collisions can make operations slower.", "What happens when two keys have the same hash bucket?", "A collision-resolution strategy such as chaining stores multiple entries in that bucket.", {"average lookup": "O(1)", "worst lookup": "O(n)"}),
    ("dsa", "hash sets", "A hash set stores unique values and supports membership checks using hashing. It is useful for deduplication and tracking whether a value has already appeared.", "How can a set help detect duplicates in a list?", "Scan values and report a duplicate whenever a value is already in the seen set.", {"average membership": "O(1)", "space": "O(n)"}),
    ("dsa", "binary trees", "A binary tree node has at most two children. Tree traversals visit nodes in preorder, inorder, postorder, or level order, each taking linear time.", "What is the difference between preorder and inorder traversal?", "Preorder visits node-left-right; inorder visits left-node-right.", {"traversal": "O(n)", "recursive space": "O(h)"}),
    ("dsa", "binary search trees", "A binary search tree orders values so left descendants are smaller and right descendants are larger, subject to its duplicate policy. An unbalanced tree can degrade to a chain.", "Why can a binary search tree have O(n) lookup?", "Sorted insertion can create a one-sided tree with height n.", {"balanced lookup": "O(log n)", "worst lookup": "O(n)"}),
    ("dsa", "heaps", "A binary heap is a complete tree satisfying a parent-child priority rule. It supports efficient access to the minimum or maximum and is commonly used to implement priority queues.", "How can a heap find the k largest values efficiently?", "Maintain a min-heap of size k while scanning the values.", {"peek": "O(1)", "insert/remove": "O(log n)"}),
    ("dsa", "graphs", "A graph consists of vertices and edges and may be directed, undirected, weighted, or unweighted. Adjacency lists are space-efficient for sparse graphs.", "When would you use an adjacency matrix instead of a list?", "Use a matrix when the graph is dense or constant-time edge-existence checks matter.", {"adjacency list space": "O(V+E)", "matrix space": "O(V^2)"}),
    ("dsa", "tries", "A trie stores strings by shared prefixes. Each edge represents a character, making prefix queries efficient at the cost of memory for nodes and links.", "What query is a trie especially suited to?", "Find all dictionary words sharing a prefix such as 'interview'.", {"lookup": "O(m)", "space": "depends on stored prefixes"}),
    ("dsa", "dynamic programming", "Dynamic programming solves problems with overlapping subproblems and optimal substructure by storing results. State definition and transitions must be precise.", "How do memoization and tabulation differ?", "Memoization caches recursive results on demand; tabulation fills a table iteratively.", {"time": "number of states times transitions", "space": "number of stored states"}),
    ("dsa", "recursion", "Recursion expresses a problem through smaller instances and needs a base case plus progress toward it. Each call consumes stack space until it returns.", "What two parts must a correct recursive algorithm have?", "A terminating base case and a recursive step that moves toward that case.", {"time": "depends on recurrence", "call stack": "O(recursion depth)"}),
    ("dsa", "backtracking", "Backtracking explores choices incrementally and reverses a choice when it cannot lead to a desired solution. Pruning avoids exploring branches that violate constraints.", "How does backtracking differ from brute force enumeration?", "Both may explore many candidates, but backtracking prunes partial candidates as soon as they cannot succeed.", {"time": "often exponential", "space": "often O(depth) excluding output"}),
    ("dsa", "sliding window", "A sliding window maintains information for a contiguous range while moving its boundaries. It often reduces repeated work from quadratic to linear time.", "When is a sliding window useful?", "Use it for contiguous subarray or substring constraints that can be updated as endpoints move.", {"typical time": "O(n)", "space": "problem-dependent"}),
    ("dsa", "two pointers", "Two pointers scan a sequence using two indices, often from opposite ends or at different speeds. Sorted input can make pair-search decisions possible without nested loops.", "How can two pointers find a target sum in a sorted array?", "Move the left pointer right when the sum is too small and the right pointer left when it is too large.", {"time": "O(n)", "extra space": "O(1)"}),
    ("dsa", "binary search", "Binary search repeatedly halves a sorted search interval by comparing the target with its middle value. Its correctness depends on a monotonic discard rule.", "What prerequisite does ordinary binary search require?", "The data or predicate must be ordered so one comparison safely eliminates half the remaining candidates.", {"time": "O(log n)", "iterative space": "O(1)"}),
    ("dsa", "sorting", "Sorting rearranges values according to an ordering. Algorithm choice depends on input size, stability, memory limits, and whether data is already partially ordered.", "What does it mean for a sorting algorithm to be stable?", "Equal-key elements retain their original relative order after sorting.", {"comparison lower bound": "O(n log n)", "space": "algorithm-dependent"}),
    ("dsa", "greedy algorithms", "A greedy algorithm chooses a locally attractive option at each step. It is correct only when the problem has a property proving local choices can lead to a global optimum.", "Why is a greedy strategy not automatically optimal?", "A locally best choice can block a better overall solution unless a proof justifies the choice property.", {"time": "problem-dependent", "proof required": "yes"}),
    ("programming", "Python", "Python emphasizes readable syntax and a large standard library. Its dynamic typing is convenient, while runtime type errors and interpreter overhead are considerations for production systems.", "When might Python be a good choice for an interview solution?", "It supports concise data-structure code and rapid iteration; state complexity and edge cases clearly.", {"dictionary lookup average": "O(1)", "list append amortized": "O(1)"}),
    ("programming", "Java", "Java is a statically typed, garbage-collected language with a mature virtual machine and extensive libraries. Interfaces and classes make contracts explicit.", "How does an interface help in Java design?", "It defines operations a class promises to implement, allowing callers to depend on a contract.", {"array access": "O(1)", "memory management": "garbage collected"}),
    ("programming", "C", "C provides low-level memory and pointer control with a small runtime. This enables systems programming but makes bounds, ownership, and lifetime errors the programmer's responsibility.", "What risks come with manual memory management in C?", "Leaks, double frees, use-after-free, and buffer overflows can cause failures or vulnerabilities.", {"pointer access": "O(1)", "memory safety": "programmer-managed"}),
    ("programming", "C++", "C++ combines low-level control with abstractions such as templates, RAII, and standard containers. RAII ties resource release to object lifetime.", "How does RAII help manage resources in C++?", "A resource-owning object's destructor releases the resource automatically when its lifetime ends.", {"vector append": "amortized O(1)", "resource cleanup": "scope-based"}),
    ("cs_fundamentals", "DBMS transactions", "A database transaction groups operations into a unit with atomicity, consistency, isolation, and durability goals. Isolation level controls how concurrent transactions interact.", "What does atomicity mean in ACID?", "A transaction commits all its operations or none of them.", {"ACID": "Atomicity, Consistency, Isolation, Durability"}),
    ("cs_fundamentals", "database indexes", "An index is an auxiliary structure that speeds up selected reads by mapping search keys to rows. It consumes storage and adds work to writes that maintain it.", "Why can adding every possible database index hurt performance?", "Each index costs space and must be updated on insert, update, or delete operations.", {"lookup": "often O(log n)", "write overhead": "increases with indexes"}),
    ("cs_fundamentals", "SQL joins", "SQL joins combine rows from tables using related columns. Join type determines what happens to unmatched rows; query plans and indexes affect performance.", "How does a LEFT JOIN differ from an INNER JOIN?", "LEFT JOIN preserves every left-table row and fills right-side columns with NULL when there is no match; INNER JOIN keeps only matches.", {"common join key": "foreign key to primary key"}),
    ("cs_fundamentals", "operating system processes", "A process is a running program with its own virtual address space and operating-system resources. Processes provide isolation but communication often requires explicit mechanisms.", "Why are processes more isolated than threads in one process?", "Separate virtual address spaces prevent ordinary writes in one process from directly changing another's memory.", {"address space": "process-private by default"}),
    ("cs_fundamentals", "threads and synchronization", "Threads share process memory and can execute concurrently. Locks, condition variables, or atomic operations coordinate access to shared state and prevent races.", "What is a race condition?", "The result depends on an uncontrolled ordering of concurrent operations on shared state.", {"shared memory": "within process", "coordination": "synchronization primitives"}),
    ("cs_fundamentals", "deadlocks", "A deadlock is a set of tasks waiting forever for resources held by one another. The classic necessary conditions are mutual exclusion, hold and wait, no preemption, and circular wait.", "Name the four classic deadlock conditions.", "Mutual exclusion, hold and wait, no preemption, and circular wait.", {"necessary conditions": "4"}),
    ("cs_fundamentals", "computer networks TCP and UDP", "TCP provides a reliable ordered byte stream with connection management and congestion control. UDP sends independent datagrams with less protocol overhead but no built-in delivery guarantee.", "When might an application choose UDP over TCP?", "Latency-sensitive voice or gaming may prefer UDP and handle loss at the application layer.", {"TCP": "reliable ordered stream", "UDP": "datagrams without delivery guarantee"}),
    ("cs_fundamentals", "HTTP and HTTPS", "HTTP is an application protocol for request-response communication. HTTPS carries HTTP over TLS, which provides encryption and server authentication when certificates are validated correctly.", "What does TLS add when using HTTPS?", "It encrypts transport and authenticates the server, helping protect integrity and confidentiality.", {"HTTP": "application layer", "HTTPS": "HTTP over TLS"}),
    ("cs_fundamentals", "object-oriented encapsulation", "Encapsulation groups state with operations and controls access to internal representation. A clear public interface lets implementation change without breaking callers.", "How does encapsulation reduce coupling?", "Callers use a stable public contract rather than depending on internal data layout.", {"goal": "hide implementation details"}),
    ("cs_fundamentals", "object-oriented polymorphism", "Polymorphism lets code use a shared interface while the concrete object supplies behavior. It supports extensibility when implementations honor the contract.", "Give an example of runtime polymorphism.", "A list of Shape objects can call area() while each concrete shape computes its own area.", {"dispatch": "implementation selected by object type"}),
    ("cs_fundamentals", "software testing", "Unit tests check small components, integration tests check collaborating components, and end-to-end tests exercise user workflows. Tests should target observable behavior and edge cases.", "What is one trade-off between unit and end-to-end tests?", "Unit tests are fast and localized; end-to-end tests cover integration but are slower and more brittle.", {"test pyramid": "many unit, fewer integration, selected end-to-end"}),
    ("cs_fundamentals", "computer architecture cache", "Caches store copies of frequently used data closer to a processor. Locality makes caches effective, while coherence protocols or synchronization matter when multiple cores share data.", "What is temporal locality?", "Recently accessed data is likely to be accessed again soon.", {"locality types": "temporal and spatial"}),
    ("cs_fundamentals", "web APIs REST", "A REST-style API models resources and uses HTTP methods consistently. Good APIs validate input, return meaningful status codes, and evolve contracts carefully.", "What should an API return for invalid client input?", "A suitable 4xx status with a clear error body, without exposing internal details.", {"GET": "read", "POST": "create or process"}),
    ("cs_fundamentals", "cloud computing", "Cloud computing provides on-demand infrastructure and managed services. Teams trade direct control for elasticity and operational convenience while still managing security and cost.", "What is one cost risk of cloud elasticity?", "Unbounded scaling or unused resources can increase bills unless budgets and limits are monitored.", {"models": "IaaS, PaaS, SaaS"}),
    ("ai", "machine learning overfitting", "Overfitting occurs when a model fits training examples but generalizes poorly to unseen data. Separate validation data, regularization, and more representative data can help diagnose or reduce it.", "How can validation loss reveal overfitting?", "If training loss keeps falling while validation loss rises, the model may be memorizing training examples.", {"splits": "train, validation, test"}),
    ("ai", "computer vision classification", "Image classification maps an image to one or more labels. Evaluation should consider class imbalance, data leakage, and metrics beyond accuracy when error costs differ.", "Why may accuracy be misleading on an imbalanced dataset?", "A model can predict the majority class and score highly while failing on rare classes.", {"metrics": "precision, recall, F1, confusion matrix"}),
    ("ai", "generative language models", "A causal language model predicts the next token from previous tokens. It can produce plausible text but may state false claims, so output needs evaluation and domain-appropriate safeguards.", "Why can fluent generated text still be incorrect?", "The model optimizes token patterns and does not guarantee that each generated claim is grounded in fact.", {"objective": "next-token prediction"}),
    ("security", "password hashing", "Passwords should be stored using a purpose-built slow password hash with a unique salt, not reversible encryption or a fast general hash. Rate limiting and multi-factor authentication provide additional protection.", "Why salt password hashes?", "A unique salt prevents identical passwords from sharing hashes and frustrates precomputed lookup tables.", {"examples": "Argon2id, scrypt, bcrypt"}),
    ("security", "SQL injection", "SQL injection occurs when untrusted input is interpreted as SQL syntax. Parameterized queries separate data from query structure and are the primary defense.", "Why are parameterized SQL queries safer than string concatenation?", "The database treats bound values as data rather than executable query syntax.", {"defense": "parameterized queries"}),
    ("interview", "STAR behavioral answers", "STAR structures a behavioral answer around Situation, Task, Action, and Result. Strong answers state the candidate's specific contribution and use concrete outcomes without overstating evidence.", "How should you answer a question about a difficult project challenge?", "Describe the situation and task, explain your actions, state the result, and briefly reflect on what you learned.", {"structure": "Situation, Task, Action, Result"}),
    ("interview", "project explanation", "A concise project explanation covers the problem, your role, design choices, one trade-off, measurable outcome, and next improvement. Be precise about which work you personally completed.", "How can you show ownership when explaining a team project?", "Separate your own decisions and implementation from the team's overall result, then explain evidence of impact.", {"structure": "problem, role, design, result, reflection"}),
    ("interview", "resume project questions", "Resume-based questions should connect directly to listed technologies and decisions. Ask about architecture, alternatives, testing, failures, the candidate's contribution, and measurable impact.", "What follow-up questions fit a Flask and OpenCV attendance project?", "Ask why Flask and OpenCV were chosen, how recognition errors were tested, where records were stored, and what the candidate personally built.", {"focus": "specific claims and trade-offs"}),
    ("resume", "resume skills extraction", "A resume skills section should list technologies the candidate can discuss with concrete examples. Automated extraction should treat matches as candidates for review because abbreviations and context can create false positives.", "How should an automated system handle a detected skill keyword?", "Return it as an extracted candidate and let the student verify proficiency and add evidence from projects or experience.", {"method": "regex candidate extraction plus human review"}),
    ("resume", "resume project evidence", "A strong project bullet states an action, the technical method, and an outcome. Metrics should be truthful and explain how they were measured; do not invent impact numbers.", "How can a student improve a vague resume project bullet?", "Name the feature implemented, relevant tools, and a verified outcome or test result, while keeping the claim accurate.", {"template": "action + method + verified result"}),
    ("interview", "internship interview preparation", "Internship interviews often assess fundamentals, learning approach, communication, and evidence of initiative. Candidates should be ready to explain coursework and projects at the depth of their own contribution.", "What should a fresher do when they do not know an interview answer?", "State what you know, explain how you would reason about the unknown part, and ask a clarifying question rather than bluffing.", {"focus": "fundamentals and learning ability"}),
    ("dsa", "breadth first search", "Breadth-first search explores a graph in layers using a queue. In an unweighted graph it finds a shortest path by number of edges from a source.", "Why does BFS find shortest paths in an unweighted graph?", "Mark a vertex when enqueuing it so it is not added repeatedly.", {"time": "O(V+E)", "space": "O(V)"}),
    ("dsa", "depth first search", "Depth-first search follows one branch before backtracking and can be implemented recursively or with an explicit stack. It supports connectivity checks, cycle analysis, and topological algorithms.", "What is a risk of recursive DFS on a very deep graph?", "The call stack may overflow; an explicit stack can avoid recursive call depth.", {"time": "O(V+E)", "space": "O(V)"}),
    ("dsa", "Dijkstra shortest path", "Dijkstra's algorithm uses a priority queue to repeatedly settle the closest unsettled vertex. It requires non-negative edge weights; negative edges need another method.", "Why does Dijkstra fail with negative edge weights?", "A later negative edge can create a shorter path to a vertex already considered settled.", {"binary heap": "O((V+E) log V)", "weight requirement": "non-negative"}),
    ("dsa", "topological sorting", "A topological order places every directed acyclic graph vertex before its outgoing neighbors. It exists only for a DAG and can be found with indegree tracking or DFS postorder.", "What does it mean if Kahn's algorithm processes fewer than V vertices?", "The directed graph contains a cycle, so no topological ordering exists.", {"time": "O(V+E)", "graph requirement": "directed acyclic"}),
    ("dsa", "union find", "Disjoint-set union tracks connected components using parent links, path compression, and union by rank or size. These heuristics make operations nearly constant amortized time.", "Where is union find useful?", "It can maintain connected components while processing edges, such as in Kruskal's spanning-tree algorithm.", {"amortized operation": "O(alpha(n))"}),
    ("dsa", "merge sort", "Merge sort divides an input into halves, sorts each half, and merges the sorted results. It has predictable n log n time and typically needs additional memory for arrays.", "What is a notable trade-off of merge sort on arrays?", "It guarantees O(n log n) time and can be stable, but usually allocates O(n) auxiliary space.", {"time": "O(n log n)", "array auxiliary space": "O(n)"}),
    ("dsa", "quicksort", "Quicksort partitions elements around a pivot and sorts the partitions recursively. Expected time is n log n with balanced pivots, but poor pivot choices can cause quadratic time.", "Why does randomized pivot selection help quicksort?", "It reduces the chance that input order repeatedly produces highly unbalanced partitions.", {"average time": "O(n log n)", "worst time": "O(n^2)"}),
    ("dsa", "amortized analysis", "Amortized analysis bounds the average cost per operation over a sequence of operations, even when individual operations vary substantially. It is not a probabilistic assumption about input distribution.", "Why is dynamic-array append amortized O(1)?", "Occasional resizing costs O(n), but geometric capacity growth spreads that cost across many constant-time appends.", {"dynamic array append": "amortized O(1)"}),
    ("cs_fundamentals", "database normalization", "Normalization organizes relational tables to reduce unwanted duplication and update anomalies. Denormalization can be chosen deliberately for read performance when trade-offs are understood.", "What update anomaly can normalization help prevent?", "The same fact stored in multiple rows may be updated inconsistently if one copy is missed.", {"common forms": "1NF, 2NF, 3NF"}),
    ("cs_fundamentals", "transaction isolation levels", "Isolation levels define which effects of concurrent transactions are visible. Stronger isolation can prevent anomalies but may reduce concurrency or increase retries.", "What is a non-repeatable read?", "A transaction reads the same row twice and sees different committed values because another transaction updated it.", {"examples": "read uncommitted, read committed, repeatable read, serializable"}),
    ("cs_fundamentals", "virtual memory", "Virtual memory gives processes a virtual address space translated to physical memory by hardware and operating-system page tables. It supports isolation and lets memory be managed in pages.", "What is a page fault?", "It occurs when an access needs a page that is not currently mapped as resident, requiring operating-system handling.", {"translation": "virtual address to physical address"}),
    ("cs_fundamentals", "CPU scheduling", "A CPU scheduler chooses which runnable task should execute. Scheduling policies balance throughput, response time, fairness, and context-switch overhead.", "What is a possible drawback of round-robin scheduling?", "A very small time quantum causes frequent context switches; a very large one harms interactive response time.", {"policy": "time-sliced fairness"}),
    ("cs_fundamentals", "DNS resolution", "DNS maps names to records such as IP addresses through a distributed hierarchy and caching. Cached answers reduce latency but may remain stale until their TTL expires.", "Why can a DNS change take time to appear everywhere?", "Resolvers may cache the old answer until its time-to-live expires.", {"records": "A, AAAA, CNAME, MX"}),
    ("cs_fundamentals", "TLS certificates", "TLS certificates bind an identity to a public key through a certificate authority chain. Clients validate the chain, hostname, validity dates, and trust configuration.", "What does hostname validation protect against in TLS?", "It checks that the presented certificate is for the server name the client intended to contact.", {"checks": "chain, hostname, dates, trust"}),
    ("cs_fundamentals", "load balancing", "A load balancer distributes requests across backend instances and can perform health checks. It improves capacity and availability but does not by itself make application state durable.", "Why should a load balancer health-check backends?", "It can stop routing new traffic to an instance that is failing or unavailable.", {"strategies": "round robin, least connections, weighted"}),
    ("cs_fundamentals", "caching strategies", "A cache stores reusable results to reduce latency or backend work. Cache invalidation, expiration, consistency, and cache stampedes are important design concerns.", "What is a cache stampede?", "Many concurrent requests recompute the same expired value, causing a burst of backend work.", {"policies": "TTL, LRU, write-through"}),
    ("ai", "precision and recall", "Precision measures how many predicted positives are correct; recall measures how many actual positives are found. The appropriate trade-off depends on the cost of false positives and false negatives.", "In screening a rare security issue, when might recall matter more than precision?", "If missing an issue is very costly, it may be preferable to flag more candidates for later review.", {"precision": "TP/(TP+FP)", "recall": "TP/(TP+FN)"}),
    ("ai", "training data leakage", "Data leakage occurs when training uses information unavailable at prediction time or examples from evaluation contaminate training. It can make reported metrics unrealistically optimistic.", "Why should related examples be grouped when splitting a dataset?", "Near-duplicates across splits let evaluation reward memorization rather than generalization.", {"best practice": "split by entity or topic when examples are related"}),
    ("ai", "language model evaluation", "Language-model evaluation can include held-out next-token loss, task-specific tests, factuality review, and human assessment. No single metric proves usefulness or safety.", "Why is perplexity not enough to evaluate a chatbot?", "It measures token prediction uncertainty but not factuality, helpfulness, instruction following, or user outcomes.", {"metrics": "loss, perplexity, task checks, human review"}),
    ("resume", "ATS keyword matching", "Applicant tracking keyword matching can help identify relevant resume terms, but keyword presence is not proof of skill. Candidates should connect relevant terms to projects or work they actually completed.", "Why should a resume checker avoid treating keyword matches as a proficiency score?", "A term may appear in a course or copied list without evidence that the candidate can use it.", {"approach": "match terms, show context, let user verify"}),
    ("resume", "resume parsing limitations", "Resume layouts vary in columns, tables, icons, and headings, so extracted text can be incomplete or reordered. Regex results should be presented as suggestions and checked against the source document.", "Why can a PDF resume parser miss information?", "Some PDFs store text in unusual reading order or as images that require OCR.", {"formats": "PDF, DOCX, TXT"}),
    ("resume", "resume-based question generation", "Resume-based questions should be grounded in extracted project, skill, and experience claims. The student should be able to edit the profile and remove extraction errors before practicing.", "How can an interviewer probe a listed project fairly?", "Ask what problem it solves, what the candidate personally built, one design trade-off, testing evidence, and a next improvement.", {"grounding": "use claims from the resume and confirm them with the student"}),
    ("programming", "Python two sum implementation", "A dictionary can store previously seen values and their indices. For each current value, check whether its complement has already appeared, then return the two indices.", "Implement two sum in Python and state its complexity.", "def two_sum(values, target):\n    seen = {}\n    for i, value in enumerate(values):\n        if target - value in seen:\n            return [seen[target - value], i]\n        seen[value] = i\n    return []", {"time": "O(n)", "space": "O(n)"}),
    ("programming", "Java binary search implementation", "Iterative binary search maintains inclusive low and high bounds in a sorted array. Updating one bound after each midpoint comparison prevents repeating work.", "What boundary update avoids an infinite loop in binary search?", "Use mid + 1 when the target is larger, and mid - 1 when it is smaller; compute mid as low + (high - low) / 2.", {"time": "O(log n)", "space": "O(1)"}),
    ("programming", "C array bounds", "C arrays do not perform automatic bounds checking. A loop must stop before the array length because valid indexes run from zero through length minus one.", "What is wrong with looping through index <= length in C?", "The final index equals the length and is outside the array; use index < length.", {"valid indexes": "0 through n-1"}),
    ("programming", "C++ vector and references", "A C++ vector owns a resizable contiguous sequence. References and iterators may be invalidated when growth reallocates storage, so lifetime and invalidation rules matter.", "When can a vector reallocation invalidate a reference?", "When an insertion exceeds capacity and the vector moves its elements to a new allocation.", {"index access": "O(1)", "push_back": "amortized O(1)"}),
    ("programming", "SQL grouped aggregation", "SQL GROUP BY partitions rows into groups for aggregate calculations. A selected non-aggregate expression generally needs to be grouped or otherwise valid for the SQL dialect.", "Write a query to count employees per department.", "SELECT department_id, COUNT(*) FROM employees GROUP BY department_id;", {"scan": "often O(n) without a suitable index"}),
    ("programming", "Java exception handling", "Exceptions represent exceptional control flow and should be caught at a level that can recover or add useful context. Swallowing exceptions hides failures and makes debugging harder.", "Why is an empty catch block dangerous?", "It hides the failure and lets the program continue with potentially invalid state.", {"good practice": "handle, translate, or propagate with context"}),
    ("programming", "C pointer ownership", "A C pointer refers to an address but does not itself communicate ownership. APIs should document who allocates and frees memory and how long returned pointers remain valid.", "How can a C API communicate ownership of returned memory?", "Document whether the caller must free it, whether it is borrowed, and the valid lifetime.", {"common errors": "leak, double free, use after free"}),
    ("programming", "C++ smart pointers", "C++ smart pointers express ownership: unique_ptr for exclusive ownership and shared_ptr for shared reference-counted ownership. Prefer unique ownership unless sharing is necessary.", "When should unique_ptr be preferred over shared_ptr?", "Use unique_ptr when one owner is sufficient; it avoids reference-count overhead and unclear shared lifetimes.", {"unique_ptr": "exclusive ownership", "shared_ptr": "shared ownership"}),
    ("interview", "debugging interview approach", "A structured debugging approach reproduces the issue, narrows the failing input, checks assumptions and boundaries, forms a hypothesis, and verifies a fix with regression tests.", "What should you do before changing code to fix a bug?", "Reproduce the behavior and isolate a minimal failing case so the cause can be tested.", {"steps": "reproduce, isolate, hypothesize, fix, regression test"}),
    ("interview", "system design trade-offs", "A system-design answer starts with requirements and constraints, estimates scale, proposes components, and discusses bottlenecks and trade-offs. It should distinguish assumptions from facts.", "How should you begin a system-design interview question?", "Clarify functional and non-functional requirements, expected scale, and important constraints.", {"areas": "requirements, APIs, data, scaling, reliability"}),
    ("resume", "resume measurable impact", "Resume metrics should be specific, truthful, and tied to a known measurement method. If impact cannot be quantified, describe scope, reliability, or a concrete result without inventing numbers.", "What can you write instead of an unsupported percentage improvement?", "Describe the measured workload, delivered feature, test coverage, or user-facing outcome you can substantiate.", {"rule": "never fabricate metrics"}),
    ("resume", "resume skill gap planning", "A skill gap is a difference between role requirements and evidence in a student's current profile. A practical plan prioritizes foundational gaps and produces a project or exercise that demonstrates learning.", "How should a student address a missing skill on a resume?", "Learn the concept, practice it in a small project, and add it only when there is truthful evidence of use.", {"plan": "learn, practice, demonstrate, then list accurately"}),
]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def build_corpus() -> dict[str, int]:
    base: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    instructions: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    topic_splits: dict[str, str] = {}
    question_templates = (
        "Explain {topic} for a computer science interview.",
        "What should a CS student know about {topic}?",
        "Describe the main idea behind {topic} and one trade-off.",
        "Give an interview-ready summary of {topic}.",
    )

    for topic_index, (category, topic, answer, interview_question, example, complexity) in enumerate(TOPICS):
        split = "train" if topic == "Python" else "test" if topic_index % 10 == 0 else "validation" if topic_index % 10 == 5 else "train"
        topic_splits[topic.casefold()] = split
        normalized_topic = topic.replace(" ", "_").replace("/", "_")
        for variant, question_template in enumerate(question_templates, start=1):
            base[split].append(
                {
                    "id": f"curriculum_{normalized_topic}_{variant:02d}",
                    "category": category,
                    "topic": topic,
                    "question": question_template.format(topic=topic),
                    "answer": answer,
                    "difficulty": "beginner" if variant == 1 else "intermediate",
                    "question_type": "theory",
                    "seniority": "fresher" if variant == 1 else "junior",
                    "example": example,
                    "complexity": complexity,
                    "follow_up_question": interview_question,
                    "metadata": {"tags": [category, topic], "source": "curated_educational_seed"},
                }
            )
        if split != "test":
            instruction_examples = [
                (
                    "explain",
                    f"Explain {topic} in clear language for a CS student preparing for interviews.",
                    answer,
                ),
                (
                    "example",
                    f"Give a practical example that helps me understand {topic}.",
                    f"Example: {example} Core idea: {answer}",
                ),
                (
                    "mock_interview",
                    f"Ask me an interview question about {topic} and provide a concise answer guide.",
                    f"Question: {interview_question} Answer guide: {answer}",
                ),
                (
                    "complexity",
                    f"Summarize the key complexity or constraints for {topic}.",
                    f"Key properties: {json.dumps(complexity, sort_keys=True)}. {answer}",
                ),
            ]
            instructions[split].extend(
                {
                    "id": f"sft_{task}_{normalized_topic}",
                    "instruction": instruction,
                    "response": response,
                    "category": task,
                    "metadata": {"tags": [category, topic], "source": "curated_educational_seed"},
                }
                for task, instruction, response in instruction_examples
            )

    structured_records: list[dict[str, Any]] = []
    for source_name in (
        "python_fundamentals.jsonl",
        "coding_questions.jsonl",
        "cross_subject_questions.jsonl",
        "full_stack_developer_questions.jsonl",
    ):
        source_path = DATA_DIR / source_name
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            topic = record.get("topic")
            question = record.get("question")
            if not all(isinstance(value, str) and value.strip() for value in (record_id, topic, question)):
                raise ValueError(f"{source_path}:{line_number}: id, topic, and question must be non-empty strings")
            subject = str(record.get("subject", "programming"))
            record_split = topic_splits.get(topic.casefold(), "train")
            difficulty = {"fresher": "beginner", "junior": "intermediate", "senior": "advanced"}.get(
                str(record.get("seniority", "fresher")).lower(), "beginner"
            )
            answer = (
                record.get("simple_definition")
                or record.get("explanation")
                or record.get("answer_guide")
                or record.get("book_definition")
                or record.get("secondary")
            )
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"{source_path}:{line_number}: record needs a definition or explanation")

            base_record = {
                **record,
                "id": record_id,
                "category": subject,
                "topic": topic,
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "metadata": record.get("metadata", {"tags": [subject, topic], "source": source_name}),
            }
            if record.get("time_complexity") or record.get("space_complexity"):
                base_record["complexity"] = {
                    key: record[key]
                    for key in ("time_complexity", "space_complexity")
                    if record.get(key)
                }
            base[record_split].append(base_record)
            structured_records.append(record)

            if record_split == "test":
                continue

            if record.get("job_role"):
                instructions[record_split].append({
                    "id": f"sft_{record_id}_answer_guide",
                    "instruction": f"Answer this {record.get('seniority', 'fresher')}-level {record.get('job_role')} interview question about {topic}: {question}",
                    "response": str(record.get("answer_guide", answer)),
                    "category": "mock_interview",
                    "metadata": {"tags": [subject, topic, *record.get("skills", [])], "source": source_name},
                })

            answer_variants = (
                ("book_definition", "Give a formal textbook definition"),
                ("simple_definition", "Explain this in simple language"),
                ("secondary", "Give a concise alternate explanation"),
            )
            for field, task in answer_variants:
                variant_answer = record.get(field)
                if isinstance(variant_answer, str) and variant_answer.strip():
                    instructions[record_split].append({
                        "id": f"sft_{record_id}_{field}",
                        "instruction": f"{task}: {question}",
                        "response": variant_answer,
                        "category": str(record.get("question_type", "explain")),
                        "metadata": {"tags": [subject, topic, field], "source": source_name},
                    })

            if record.get("question_type") == "coding":
                response_parts = [str(record.get("explanation", ""))]
                if record.get("code"):
                    response_parts.append(f"Code ({record.get('language', 'text')}):\n{record['code']}")
                if record.get("expected_output"):
                    response_parts.append(f"Expected output: {record['expected_output']}")
                if record.get("time_complexity"):
                    response_parts.append(f"Time complexity: {record['time_complexity']}")
                if record.get("space_complexity"):
                    response_parts.append(f"Space complexity: {record['space_complexity']}")
                instructions[record_split].append({
                    "id": f"sft_{record_id}_solution",
                    "instruction": question,
                    "response": "\n".join(part for part in response_parts if part),
                    "category": "coding",
                    "metadata": {"tags": [subject, topic, str(record.get("language", ""))], "source": source_name},
                })

            for follow_up in record.get("interview_questions", []):
                level = str(follow_up.get("seniority", "fresher"))
                follow_up_question = follow_up.get("question")
                follow_up_answer = follow_up.get("answer")
                if follow_up_question and follow_up_answer:
                    instructions[record_split].append({
                        "id": f"sft_{record_id}_{level}_followup",
                        "instruction": f"Answer this {level}-level interview question about {topic}: {follow_up_question}",
                        "response": follow_up_answer,
                        "category": "mock_interview",
                        "metadata": {"tags": [subject, topic, level], "source": source_name},
                    })

    for split, records in base.items():
        write_jsonl(DATA_DIR / f"{split}.jsonl", records)
    write_jsonl(DATA_DIR / "instruction_train.jsonl", instructions["train"])
    write_jsonl(DATA_DIR / "instruction_validation.jsonl", instructions["validation"])
    return {
        "topics": len(topic_splits),
        "train_records": len(base["train"]),
        "validation_records": len(base["validation"]),
        "test_records": len(base["test"]),
        "instruction_train_records": len(instructions["train"]),
        "instruction_validation_records": len(instructions["validation"]),
    }


if __name__ == "__main__":
    print(json.dumps(build_corpus(), indent=2))
