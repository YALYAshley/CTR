# ***splitjson***.*py*

### 作用：

将一个大的json文件，根据图片名，分成多个json文件，即一张图一个json文件。每个新生成的json文件名为原json文件中每张图的名称。

### 使用方法：

例如，有目录

```
stop
	center
		1.json
	left
	    2.json
	right
	    3.json
```

1. 首先将该文件与stop文件夹置于同一目录下，修改代码中的path的赋值为

   ```
   path = 'stop'
   ```

2. 然后在当前目录下运行改代码，新产生的json文件存储在原json文件所在目录下；

3. 最后将原有的1.json、2.json、3.json手动删除即可。

# *rename2.py*

### 作用：

按照一定的命名规则，对图片批量重命名。并且将每个图片重复四次，即共有五张一样的图片。

**本例中的重命名规则如下：**

​	原图片的名称为阿拉伯数字，如100.jpg。（图片需按数字大小排列）

​	重命名后的格式为：

```
'固定前缀'+'图片序号'+'_'+'重复图片的编号'
```

​	例如，20211228001_01.jpg        其中，20211228为固定前缀，001为图片序号，01为重复图片的编号。

### 使用方法：

例如，有目录

```
stop
	path1
		100.jpg
		101.jpg
		102.jpg
	path2
		100.jpg
		101.jpg
		102.jpg
	path3
		100.jpg
		101.jpg
		102.jpg
```

1. 首先在stop文件夹下新建三个文件夹，如new_folder1,new_folder2,new_folder3，如下所示：

   ```
   stop
   	path1
   		100.jpg
   		101.jpg
   		102.jpg
   	path2
   		100.jpg
   		101.jpg
   		102.jpg
   	path3
   		100.jpg
   		101.jpg
   		102.jpg
   	new_folder1
   	new_folder2
   	new_folder3
   ```

2. 然后修改代码中的对应文件夹名称，如下图所示：

   ![image-20211228202806175](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20211228202806175.png)

3. 接着修改重命名的图片的前缀和一个文件夹中原图片名从哪一个阿拉伯数字开始，如下图所示：

   ![image-20211228203653495](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20211228203653495.png)

   注意本例中三个path下的图片名是一样的，所以只需要一个start_num即可；

4. 最后在path1同级目录下运行代码，重复且重命名后的文件存储在对应的new_folder中；

5. 最后的最后，记得删除path1、path2和path3文件夹。

   

# *repath2.py*

### **作用：**

批量对json文件中的in_name进行重命名。

### 使用方法：

​	repath2.py文件中，data文件夹里面装的是直接生成的.json文件，new_data文件里面装的是更改过路径等内容的.json文件，这里json文件夹实际上就是代码中的new_data文件。
使用时，repath2.py文件与data文件、new_data文件要处于同一目录下，初始时new_data文件要自己新建，包括新建其下一级目录中的三个视角文件夹。

（data和new_data的名称可自己命名）

![image-20211228205833569](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20211228205833569.png)