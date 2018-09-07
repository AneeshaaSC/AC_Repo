class MLtask_2 implements Runnable
{
	String thread_name;
	int thread_id;
	
	MLtask_2(String n,int i)
	{
		this.thread_name=n;
		this.thread_id=i;
	}
	public void run()
	{
		for(int i=0;i<10;i++)
		{
			try
			{
			System.out.println("Thread name:"+thread_name+"\n"+"Thread ID:"+thread_id+" Iteration:"+i);
			Thread.sleep(500);
			}
			catch(InterruptedException ie)
			{
			System.out.println("I'm trying to get some damn sleep");
			}
			}
	}
}

public class ans2_variant {
	public static void main(String args[])
	{
		MLtask_2 m1=new MLtask_2("thread-1",101);
		Thread t1=new Thread(m1);
		t1.start();
		MLtask_2 m2=new MLtask_2("thread-2",201);
		Thread t2=new Thread(m2);
		t2.start();
		
	}

}
