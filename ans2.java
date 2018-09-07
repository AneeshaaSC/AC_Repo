class MLtask extends Thread
{
	public void run()
	{
		for(int i=0;i<10;i++)
		{
			try
			{
			System.out.println("Thread name:"+getName()+"\n"+"Thread ID:"+getId()+" Iteration:"+i);
			Thread.sleep(500);
			}
			catch(InterruptedException ie)
			{
			System.out.println("I'm trying to get some damn sleep");
			}
			}
	}
}

public class ans2 {
	public static void main(String args[])
	{
		System.out.println("Creating thread objects");
		MLtask t1=new MLtask();
		MLtask t2=new MLtask();
		System.out.println("Start multi-threaded task objects");
		t1.start();
		t2.start();
		
	}

}
