"""
Double priority queue implementation that keeps an arbitrary number of
largest and smallest members
"""

class DoublePriority:
    def __init__(self, num_smallest, num_largest, dtype=tuple):
        assert(dtype in [tuple, bool, int, float])
        self.dtype = dtype
        self.num_smallest = num_smallest
        self.num_largest = num_largest
        self.maxqueue = []
        self.minqueue = []
        
    # Insert an element into the double priority queue
    # Element is inserted into both the max and min queues
    def insert(self, element):
        if self.num_smallest > 0:
            self.minqueue.append(element)
            self.minqueue.sort()
            if (len(self.minqueue) > self.num_smallest):
                self.minqueue = self.minqueue[:-1]
        
        if self.num_largest > 0:
            self.maxqueue.append(-element if self.dtype != tuple else (-element[0], element))
            self.maxqueue.sort()
            if (len(self.maxqueue) > self.num_largest):
                self.maxqueue = self.maxqueue[:-1]
            
    def getlargest(self):
        # Undo negation
        for i in range(len(self.maxqueue)):
            item = self.maxqueue[i]
            self.maxqueue[i] = -item if self.dtype != tuple else item[1]
        return self.maxqueue
    
    def getsmallest(self):
        return self.minqueue