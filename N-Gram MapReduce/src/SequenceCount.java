
import java.io.IOException; 
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration; 
import org.apache.hadoop.fs.FileSystem; 
import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable; 
import org.apache.hadoop.io.LongWritable; 
import org.apache.hadoop.io.Text; 
import org.apache.hadoop.mapreduce.Job; 
import org.apache.hadoop.mapreduce.Mapper; 
import org.apache.hadoop.mapreduce.Reducer; 
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat; 
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

	public class SequenceCount {
	
	/** Utility Function to remove punctuation and convert all string to lowercase **/
	private static String removePunctuation(String temp){
			
		temp = temp.replaceAll("[^a-zA-Z\\s]", "").toLowerCase();
		return(temp);
	}

	/** Mapp Function for Sequence Count **/
	public static class SequenceCountMapper extends Mapper < LongWritable, Text, Text, IntWritable >
	{

		private final static IntWritable one = new IntWritable( 1); 
		private Text words = new Text();
		List contents = new ArrayList();

	
	public void map( LongWritable key, Text values, Context context) throws IOException, 
	InterruptedException {
		
		StringTokenizer st = new StringTokenizer(removePunctuation(values.toString()), " ");
		
		while(st.hasMoreTokens()){
			contents.add(st.nextToken());
			}
		}
	
	/** Overriding cleanup method to allow for continuous grouping by sequences **/
	@Override
	protected void cleanup(Context context)throws IOException, InterruptedException {
		int sequence = 5;
	
		StringBuffer sb = new StringBuffer("");
		for(int i = 0; i < contents.size() - sequence; i++){
			int counter = i;
			for(int j=0; j < sequence; j++){
				if(j>0)
				{

					sb = sb.append(" ");
					sb = sb.append(contents.get(counter));
				}
				else{
					sb = sb.append(contents.get(counter));
				}
				counter++;
			}
			words.set(sb.toString());
			sb = new StringBuffer("");
			context.write(words, one);
			}
		}
	}
	
	/** Reducer Function for Sequence Count **/
	public static class SequenceCountReducer extends Reducer < Text, IntWritable, Text, IntWritable > { 
	private IntWritable result = new IntWritable(); 
	@Override 
	public void reduce( Text key, Iterable < IntWritable > values, Context context ) throws IOException, InterruptedException { 
		int sum = 0; 
		for (IntWritable val : values) { 
			sum += val.get(); 
		} 
		result.set(sum); 
		context.write(key, result); 
		} 
	}

	/** Driver **/
	public static void main( String[] args) throws Exception { 
		Configuration conf = new Configuration();
		Job job = Job.getInstance( conf, "sequence count");
	
		job.setJarByClass(SequenceCount.class);
	
		FileInputFormat.addInputPath( job, new Path("input")); 
		FileOutputFormat.setOutputPath( job, new Path("output")); 
		job.setMapperClass( SequenceCountMapper.class); 
		job.setCombinerClass( SequenceCountReducer.class); 
		job.setReducerClass( SequenceCountReducer.class);
	
		job.setOutputKeyClass( Text.class); 
		job.setOutputValueClass( IntWritable.class);
	
		System.exit( job.waitForCompletion( true) ? 0 : 1); 
	} 
	}

