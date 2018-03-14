import random,csv,math
import numpy as np
from matplotlib import pyplot as ply

#iris_data="C:\\Users\\prach\\Desktop\\New folder\\UTA Ms\\Fall'17\\Machine Learning\\Project_1\\1001234789_Prachi_Goel\\iris.csv"

def actual_del(reader):
    class1=[]
    class2=[]
    class3=[]
    actual_cluster=[]
    with open(reader,'r') as csvread:
        data=csv.reader(csvread)
        data=list(data)
        for i in range (len(data)):
            for j in range(len(data[i])-1):
                data[i][j]=float(data[i][j])
    for i in range(len(data)):
        if data[i][4]=='Iris-setosa':
            class1.append(data[i][:4])
        elif data[i][4]=='Iris-versicolor':
            class2.append(data[i][:4])
        elif data[i][4]=='Iris-virginica':
            class3.append(data[i][:4])
    actual_cluster.append(class1)
    actual_cluster.append(class2)
    actual_cluster.append(class3)
    return actual_cluster

def importing_data(file):
    with open (file,'r') as csvfile:
        dataset=csv.reader(csvfile)
        dataset=list(dataset)
        for x in range(len(dataset)):
            dataset[x] = dataset[x][:4]
            for y in range(len(dataset[0])):
                dataset[x][y]=float(dataset[x][y])
    return dataset

def initial_centeroids(dataset,K):
    number_of_clusters=K
    initial_centeroids_of_clusters=[]
    rand = random.sample(range(len(dataset)),number_of_clusters*2)
    for i in rand:
            initial_centeroids_of_clusters.append(dataset[i])
    return initial_centeroids_of_clusters

def evaluating_new_centroid(clusters):
    temp=np.array(clusters)
    new_centroid=[]
    for i in range(len(temp)):
        x=(np.average(temp[i],axis=0))
        y=x.tolist()
        new_centroid.append(y)
    return (new_centroid)

def euclidian_distance(instance,centroid):
    summation=0
    for attribute in range(len(instance)):
        summation+=(instance[attribute]-centroid[attribute])**2
    euclidian=math.sqrt(summation)
    return euclidian

def value_K(dataset):
    summation = []
    cluster_range = [i for i in range(1,8)]
    for i in range(1,8):
        initial_centeroid_Kmeans = initial_centeroids(dataset, K=i)
        cluster = kMeans(dataset, initial_centeroid_Kmeans, K=i)
        summation.append(cluster_validity(cluster[0], cluster[1]))
    ply.plot(cluster_range, summation ,'bo', cluster_range, summation, 'k')
    ply.xlabel("K: Number of Cluster ")
    ply.ylabel("Square Sum Error: Distortion")
    ply.title("The Elbow method showing the optimal value of K")
    ply.annotate('Elbow point in the graph', xy=(cluster_range[2], summation[2] + 8), xytext=(cluster_range[2], summation[2] + 100),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    ply.show()
    print ("Using elbow method the value of K=3")


def cluster_validity(cluster,centroid):
    summation=0
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            summation+=(euclidian_distance(cluster[i][j],centroid[i])**2)
    return (summation)

def kMeans(dataset,initial_centroid,K):
    n=len(initial_centroid)
    new_centroid=initial_centroid[0:int(n/2)]
    prev_centroid=initial_centroid[int(n/2):n]
    iteration=0
    while len([p for p in (new_centroid) if p not in prev_centroid])!=0 and iteration<100:
        clusters=[[] for i in range (0,K)]
        for i in range (len(dataset)):
            euclidian_metric=[]
            for j in range (len(new_centroid)):
                euclidian_metric.append(euclidian_distance(dataset[i],new_centroid[j]))
            cluster_value=np.argmin(np.array(euclidian_metric))
            clusters[cluster_value].append(dataset[i])
        prev_centroid=new_centroid
        new_centroid=evaluating_new_centroid(clusters)
        iteration+=1
    print("KMeans when K is",K,"converges at iteration:", iteration + 1)
    return clusters,new_centroid

def scatter_plot(clusters,K,centroid):
    x=[]
    y=[]
    center_x=[]
    center_y=[]
    center_label=[]
    labels=[]
    with open("./Result.csv",'w') as result_file:
        Cluster_number=0
        wrt=csv.writer(result_file)
        for i in clusters:
            Cluster_number+=1
            wrt.writerow("Observations in Cluster: "+str(Cluster_number))
            for j in i:
                wrt.writerow(j)

    for i in range (len(centroid)):
        center_x.append(centroid[i][0])
        center_y.append(centroid[i][3])
        center_label.append(i)
    for i in range (len(clusters)):
        for j in range(len(clusters[i])):
            x.append(clusters[i][j][0])
            y.append(clusters[i][j][3])
            labels.append(i)
    label_color_map = {0: 'r',
                       1: 'g',
                       2: 'b',
                       3: 'n',
                       4: 'k',
                       5: 'c',
                       6: 'w'}
    label_color = [label_color_map[l] for l in labels]
    center_label_color = [label_color_map[l] for l in center_label]
    x=ply.scatter(x,y,s=50,alpha=0.7,c=label_color,label='Data Points')
    y=ply.scatter(center_x,center_y,s=80,alpha=1, c=center_label_color, marker='*',label='Centroid')
    ply.title('Scatter plot for KMeans when K is '+str(K) )
    ply.legend(loc=2)
    ply.show()
    return x,y



def accuracy(cluster,actual_data):
    clusters=['Iris-setosa','Iris-versicolor','Iris-virginica']
    accu=[]
    for i in range (len(cluster)):
        add = []
        for j in range(len(actual_data)):
            add.append(len([p for p in cluster[i] if p not in actual_data[j]])+len([p for p in actual_data[j] if p not in cluster[i]]))
        if np.argmin(np.array(add))==0:
            classification='Iris-setosa'
        elif np.argmin(np.array(add))==1:
            classification='Iris-versicolor'
        elif np.argmin(np.array(add))==2:
            classification='Iris-virginica'
        print("cluster",i,"is equivalent to class",classification,"with accuracy of ",100-((float(add[np.argmin(np.array(add))])/50)*100),"% accuracy")
        accu.append(100 - ((float(add[np.argmin(np.array(add))]) / 50) * 100))
    ply.bar(clusters,accu)
    ply.xlabel('Classes/Cluster')
    ply.ylabel('Accuracy')
    ply.title('Accuracy per Cluster')
    for i in range(len(clusters)):
        ply.text(clusters[i],accu[i]+1,accu[i],)
    ply.show()

if __name__ == '__main__':
    iris_data=input("Please enter the Iris data file location(kindly ensure data is in csv file format with row as an observation)")
    #Result=input("Please enter the file location for output file")
    dataset=importing_data(iris_data)
    initial_centeroid_Kmeans = initial_centeroids(dataset, K=3)
    value_K(dataset)
    cluster=kMeans(dataset,initial_centeroid_Kmeans,K=3)
    scatter_plot(cluster[0],3,cluster[1])
    actual_data=actual_del(iris_data)
    accuracy(cluster[0],actual_data)


