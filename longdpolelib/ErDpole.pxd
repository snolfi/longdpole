cdef extern from "utilities.cpp":
    pass

cdef extern from "dpole.cpp":
    pass

# Declare the class with cdef
cdef extern from "utilities.h":
    cdef cppclass RandomGenerator:
        RandomGenerator() except +
        void setSeed(int seed)
        int seed()
        int getInt(int min, int max)
        double getDouble(double min, double max)
        double getGaussian(double var, double mean)

# Declare the class with cdef
cdef extern from "dpole.h":
    cdef cppclass Problem:
        Problem() except +
        double* m_state
        double m_masspole_2, m_length_2
        int m_ninputs, m_noutputs
        double m_high, m_low
        void seed(int s)
        void reset()
        double step()
        void close()
        void render()
        void copyObs(float* observation)
        void copyAct(float* action)
        void copyDone(int* done)
        void copyDobj(double* objs)
        void setLengthMass2(double length_2, double masspole_2)

