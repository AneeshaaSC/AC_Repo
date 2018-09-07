	class task1 implements Runnable
	{
		public void run()
		{
			for(int i=0;i<=5;i++)
			{
				try {
				System.out.println("task1: "+i);
				Thread.sleep(1000);}
				catch(InterruptedException ie) { }
			}
			System.out.println("Exiting thread1");
		}
	}

	class task2 implements Runnable
	{
		public void run()
		{
			for(int i=0;i<=5;i++)
			{
				try 
				{
				System.out.println("task2: "+i);
			    Thread.sleep(1000);
				}
			catch(InterruptedException ie) { }
			}
			System.out.println("Exiting thread2");
		}
	}
public class ans3 {

	public static void main(String args[]) {
		task2 t2=new task2();
		Thread THREAD2=new Thread(t2);
		
		task1 t1=new task1();
		Thread THREAD1=new Thread(t1);
		
		THREAD1.start();
		THREAD2.start();
	}
}
