
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException; 
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration; 
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem; 
import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable; 
import org.apache.hadoop.io.LongWritable; 
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text; 
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job; 
import org.apache.hadoop.mapreduce.Mapper; 
import org.apache.hadoop.mapreduce.Reducer; 
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat; 
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


/** 
 * CIS-5570 
 * Assignment 3
 * 10/14/2018
 * MapReduce One-Pass Matrix Multiplication Implementation 
 * Author: Kendra Bach
 * 
 **/
	public class MatrixMultiplication {
		
		/** Utility Function for Assigning an Identity to each Matrix and Retrieving Dimensions **/	
		private static void formatMatrix(String filename, Integer i, Integer k, Configuration conf) throws IOException{
			
			//Create variables and establish file system
			FileSystem fs = FileSystem.get(conf);
			String fileLine;
			ArrayList<String> sb = new ArrayList<String>();
			boolean delFile = true;
			
			Path newPath = new Path("output/newFile.txt");
			FSDataOutputStream newFile = fs.create(newPath);
			
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(filename))));
			
			//Read current numbers
			while((fileLine = br.readLine()) != null){
				sb.add(fileLine);
				}
			
			br.close();
			
			//For each row that is less than the total dimensions of the matrix
			//Adding k to row every time to prevent redundancy
			for(int row=0; row < (i*k); row+=k)
			{
				
				for(int col=row; col < (row+(k)); col++)
				{
					newFile.write((sb.get(col).toString()).getBytes());
					newFile.write(" ".getBytes());
				}
				newFile.write("\n".getBytes());
				
	
			}
	
			//Close all input and output to prevent leaks
			br.close();
			newFile.close();
			
			//Configure file name
			Path oldFile = new Path(filename);
			fs.delete(oldFile, delFile);
			
			//Rename new file to match old
			Path origName = new Path(filename);
			fs.rename(newPath, origName);
			
			
		}
		
	/** Utility Function for Assigning an Identity to each Matrix and Retrieving Dimensions **/	
	private static ArrayList<Integer> addMatrixIdentity(String filename, Configuration conf) throws IOException{
		
		//Create file system variable
		FileSystem fs = FileSystem.get(conf);
		boolean delFile = true;
		
		String matrixOrder = "0";
		String fileLine;
		Integer i = 0;
		Integer k = 0;
		Integer j = 0;
		
		FSDataOutputStream newFile = fs.create(new Path("input/newFile.txt"));
		
		Path origFile = new Path(filename);

		BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(origFile)));
			
		//For every line in both matrices
		//Increment the matrix count/identity 
		//  if newline is reached
		while((fileLine = br.readLine()) != null){
			if(fileLine.length() == 0){
				matrixOrder = "1";
				continue;
			}
			
			//If matrix identity is 1
			if(Integer.parseInt(matrixOrder) == 1)
			{
				
				String[] cols = fileLine.split(" ");
				k = cols.length;
				
				fileLine = matrixOrder + " " + j + " " + fileLine + "\n";
				newFile.write(fileLine.getBytes());
				
				j++;
			}
			
			//If matrix identity is 0
			if(Integer.parseInt(matrixOrder) == 0)
			{
				fileLine = matrixOrder + " " + i + " " + fileLine + "\n";
				newFile.write(fileLine.getBytes());
			
				i++;
			}

		}

		//Close all input and output
		br.close();
		newFile.close();
		
		//Establish name for new file
		//File oldFile = new File(filename);
		//oldFile.delete();
		fs.delete(new Path("input/matrix.txt"), delFile);
		
		//Rename new file to match that of the original
		//File origName = new File(filename);
		fs.rename(new Path("input/newFile.txt"), new Path("input/matrix.txt"));
		
		//Delete any backups that were made
		if(fs.exists(new Path("input/matrix.txt~")))
		{
			fs.delete(new Path("input/matrix.txt~"), delFile);
		}
		
		//Retrieve the indices for current matrices
		ArrayList<Integer> indices = new ArrayList<Integer>();
		indices.add(i);
		indices.add(j);
		indices.add(k);
		
		return(indices);
		
	}

	/** Map Function for Matrix Multiplication **/
	public static class MatrixMultiplicationMapper extends Mapper <LongWritable, Text, Text, Text >
	{

		String matrixNum = null;
		MapWritable keyMap = new MapWritable();
		ArrayList<Integer> valueList = new ArrayList<Integer>(); 
		HashMap<Integer, Integer> rowMap = new HashMap<Integer, Integer>();
		StringBuilder sbKeys = new StringBuilder();
		StringBuilder sbVals = new StringBuilder();
		
		
	public void map(LongWritable key, Text values, Context context) throws IOException, 
	InterruptedException {
		
		//Retrieve config variables
		Configuration config = context.getConfiguration();
		Integer i = Integer.parseInt(config.get("i"));
		Integer j = Integer.parseInt(config.get("j"));
		Integer k = Integer.parseInt(config.get("k"));
		

		StringTokenizer st = new StringTokenizer(values.toString(), " ");
		

		if(st.hasMoreTokens()){
			matrixNum = st.nextToken().toString();
			
			//For rows from the first matrix 
			if(Integer.parseInt(matrixNum) == 0)
			{
				//For each column less than the largest column + 1 in first matrix (j)
				Integer row = Integer.parseInt(st.nextToken().toString());
					for(int cols = 0; cols < j; cols++)
					{	
						//For each column less than the resulting total amount of columns + 1 (k)
						String mVal = st.nextToken().toString();
						for(int resultCol = 0; resultCol < k; resultCol++)
						{
							
							sbKeys.append(row);
							sbKeys.append(",");
							sbKeys.append(resultCol);
							
							sbVals.append(matrixNum);
							sbVals.append(',');
							sbVals.append(cols);
							sbVals.append(',');
							sbVals.append(mVal);
							
							context.write(new Text(sbKeys.toString()), new Text(sbVals.toString()));
							
							sbVals.delete(0, sbVals.length());
							sbKeys.delete(0, sbKeys.length());
							
						
						
					}
				}
			}
			
			//For rows from the second matrix
			else if (Integer.parseInt(matrixNum) == 1){
			
				//For each column less than the largest column + 1 in second matrix (k)
				Integer jCol = Integer.parseInt(st.nextToken().toString());
					for(int cols = 0; cols < k; cols++)
					{
						//For each row less than the total rows + 1 in resulting matrix (i)
						String nVal = st.nextToken().toString();
						for(int resultRow = 0; resultRow < i; resultRow++)
						{
							sbKeys.append(resultRow);
							sbKeys.append(",");
							sbKeys.append(cols);
							
							sbVals.append(matrixNum);
							sbVals.append(',');
							sbVals.append(jCol);
							sbVals.append(',');
							sbVals.append(nVal);
							
							context.write(new Text(sbKeys.toString()), new Text(sbVals.toString()));
							sbVals.delete(0, sbVals.length());
							sbKeys.delete(0, sbKeys.length());
						}
					}
				}
			}
		}
	}


	/** Reducer Function for Matrix Multiplication **/
	public static class MatrixMultiplicationReducer extends Reducer < Text, Text, Text, Text> { 

	HashMap<ArrayList<Integer>, Integer> keyValue = new HashMap<ArrayList<Integer>, Integer>();
	Integer largestCol;
	HashMap<Integer, Integer> matrixM = new HashMap<Integer, Integer>();
	HashMap<Integer, Integer> matrixN = new HashMap<Integer, Integer>();
	ArrayList<Integer> sumList = new ArrayList<Integer>();

 
	public void reduce(Text key, Iterable<Text> values, Context context ) throws IOException, InterruptedException { 
		ArrayList<Integer> vals = new ArrayList<Integer>();
		sumList = new ArrayList<Integer>();
		Integer sum = 0;
		
		//Iterate through individual array vales and convert to int
		for(Text txt : values)
		{
			String strVals = txt.toString();
			String[] strValList = strVals.split(",");
			vals.clear();
			for(int v = 0; v < strValList.length; v++)
			{
				Integer val = Integer.parseInt((strValList[v]));
				vals.add(val);
				
			}
			if(vals.size() > 2)
			{
			
			Integer matrixNum = vals.get(0);
			Integer j = vals.get(1);
			Integer val = vals.get(2);
			
			//Categorize matrix values by matrix number and j value
			if(matrixNum == 0)
			{
				matrixM.put(j, val);
	
			}
			else
			{
				matrixN.put(j, val);
			}
			}
		}
		
		//Find same j-valued values in each array and multiply
		for(Entry<Integer, Integer> entry : matrixM.entrySet())
		{
			Integer mKey = entry.getKey();
			Integer mVal = entry.getValue();
			
			Integer nVal = matrixN.get(mKey);
			
			Integer mult = mVal * nVal;
			sumList.add(mult);
		}
		
		//Sum all values in list
		for(int i = 0; i < sumList.size(); i++)
		{
			sum += sumList.get(i);
		}
		
		context.write(null, new Text(sum.toString()));
			}

		}
		

	/** Driver **/
	public static void main( String[] args) throws Exception { 
		ArrayList<Integer> dimensions = new ArrayList<Integer>();		
		String inputFilePath = "input/matrix.txt";
		String outputFilePath = "output/part-r-00000";
		
		//Format File for MapReduce 
		Configuration conf = new Configuration();
		//conf.set("fs.default.name", "hdfs://localhost:8020");
		dimensions = addMatrixIdentity(inputFilePath, conf);
		
		//Set Config Variables to be Broadcasted/Accessible Amongst all Nodes
		conf.set("i", dimensions.get(0).toString());
		conf.set("j", dimensions.get(1).toString());
		conf.set("k", dimensions.get(2).toString());
		Job job = Job.getInstance( conf, "matrix multiplication");
	
		job.setJarByClass(MatrixMultiplication.class);
		
		job.setMapOutputKeyClass(Text.class); 
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class); 
		job.setOutputValueClass(Text.class);
	
		FileInputFormat.addInputPath( job, new Path("input")); 
		FileOutputFormat.setOutputPath( job, new Path("output")); 
		
		job.setMapperClass( MatrixMultiplicationMapper.class); 
		job.setReducerClass( MatrixMultiplicationReducer.class);
		
		job.waitForCompletion( true); 
		
		//Format Final Output
		formatMatrix(outputFilePath, dimensions.get(0), dimensions.get(2), conf);
		
		
		} 
	}
	

