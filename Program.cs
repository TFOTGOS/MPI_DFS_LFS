using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MPI;

class Program
{
    static void Main(string[] args)
    {
        using (new MPI.Environment(ref args))
        {
            var comm = Communicator.world;
            int rank = comm.Rank;
            int size = comm.Size;

            int[] graphSizes = { 100000, 300000, 1000000 };
            double[,] results = new double[graphSizes.Length, 2];

            for (int i = 0; i < graphSizes.Length; i++)
            {
                int numVertices = graphSizes[i];
                int avgDegree = 4;

                List<int>[] graph = null;

                if (rank == 0)
                {
                    Console.WriteLine($"Graph size: {numVertices}");
                    graph = GenerateRandomGraph(numVertices, avgDegree);
                }

                int localStart = rank * numVertices / size;
                int localEnd = (rank + 1) * numVertices / size;
                
                graph = BroadcastGraph(graph, comm);
                comm.Barrier();

                double dfsTime = MeasureTraversalTime(graph, localStart, localEnd, isDFS: true, comm);
                double bfsTime = MeasureTraversalTime(graph, localStart, localEnd, isDFS: false, comm);

                if (rank == 0)
                {
                    results[i, 0] = dfsTime;
                    results[i, 1] = bfsTime;
                }
            }

            if (rank == 0)
            {
                Console.WriteLine("\nGraph Size\tDFS Time (s)\tBFS Time (s)");
                for (int i = 0; i < graphSizes.Length; i++)
                {
                    Console.WriteLine($"{graphSizes[i]}\t\t{results[i, 0]:0.0000}\t\t{results[i, 1]:0.0000}");
                }
            }
        }
    }

    static List<int>[] GenerateRandomGraph(int numVertices, int avgDegree)
    {
        Random rand = new Random();
        var graph = new List<int>[numVertices];
        for (int i = 0; i < numVertices; i++)
            graph[i] = new List<int>();

        for (int i = 0; i < numVertices; i++)
        {
            while (graph[i].Count < avgDegree)
            {
                int neighbor = rand.Next(numVertices);
                if (neighbor != i && !graph[i].Contains(neighbor))
                {
                    graph[i].Add(neighbor);
                    graph[neighbor].Add(i);
                }
            }
        }
        return graph;
    }

    static List<int>[] BroadcastGraph(List<int>[] graph, Intracommunicator comm)
    {
        int numVertices = graph?.Length ?? 0;

        comm.Broadcast(ref numVertices, 0);

        if (comm.Rank != 0)
        {
            graph = new List<int>[numVertices];
        }

        for (int i = 0; i < numVertices; i++)
        {
            int neighborCount = graph != null && graph[i] != null ? graph[i].Count : 0;
            comm.Broadcast(ref neighborCount, 0);

            int[] neighbors = new int[neighborCount];
            if (comm.Rank == 0)
            {
                neighbors = graph[i].ToArray();
            }
            comm.Broadcast(ref neighbors, 0);

            if (comm.Rank != 0)
            {
                graph[i] = neighbors.ToList();
            }
        }

        return graph;
    }


    static double MeasureTraversalTime(List<int>[] graph, int startVertex, int endVertex, bool isDFS, Intracommunicator comm)
    {
        bool[] visited = new bool[graph.Length];

        comm.Barrier();
        Stopwatch sw = Stopwatch.StartNew();

        for (int i = startVertex; i < endVertex; i++)
        {
            if (!visited[i])
            {
                if (isDFS)
                    DFS(graph, i, visited);
                else
                    BFS(graph, i, visited);
            }
        }

        comm.Barrier();
        sw.Stop();

        double time = sw.Elapsed.TotalSeconds;
        double totalTime = comm.Reduce(time, Operation<double>.Max, 0);

        return totalTime;
    }

    static void DFS(List<int>[] graph, int v, bool[] visited)
    {
        Stack<int> stack = new Stack<int>();
        stack.Push(v);

        while (stack.Count > 0)
        {
            int node = stack.Pop();
            if (!visited[node])
            {
                visited[node] = true;
                foreach (var neighbor in graph[node])
                {
                    if (!visited[neighbor])
                        stack.Push(neighbor);
                }
            }
        }
    }

    static void BFS(List<int>[] graph, int v, bool[] visited)
    {
        Queue<int> queue = new Queue<int>();
        visited[v] = true;
        queue.Enqueue(v);

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            foreach (var neighbor in graph[node])
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    queue.Enqueue(neighbor);
                }
            }
        }
    }
}
