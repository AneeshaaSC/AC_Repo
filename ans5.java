
public class ans5 {
	public static void main(String args[])
	{
		System.out.println("Main Thread Details:  "+"\n"+"main thread name: "+Thread.currentThread().getName()+"\n"+"main thread ID: "+Thread.currentThread().getId()+"\n"+"main thread priority: "+Thread.currentThread().getPriority()+"\n"+"main thread status: "+Thread.currentThread().getState());
		Thread t1=new Thread();
		Thread t2=new Thread();
		t1.setName("THREAD PIECE 1");
		t2.setName("THREAD PIECE 2");

		System.out.println("Thread-1 details:  "+"\n"+"thread1 name: "+t1.getName()+"\n"+"thread-1 ID: "+t1.getId()+"\n"+"thread-1 priority: "+t1.getPriority()+"\n"+"thread-1 status: "+t1.getState()+"\n"+"thread-1 thread group: "+t1.getThreadGroup());
		System.out.println("Thread-2 details:  "+"\n"+"thread2 name: "+t2.getName()+"\n"+"thread-2 ID: "+t2.getId()+"\n"+"thread-2 priority: "+t2.getPriority()+"\n"+"thread-2 status: "+t2.getState()+"\n"+"thread-2 thread group: "+t2.getThreadGroup());

		t1.setPriority(10);
		t2.setPriority(6);
		
		System.out.println("\n"+"main thread priority: "+Thread.currentThread().getPriority());
		System.out.println("Thread 1 priority:"+t1.getPriority());
		System.out.println("Thread 2 priority:"+t2.getPriority());
		
		Thread.currentThread().setPriority(10);
		
		System.out.println("\n"+"main thread priority: "+Thread.currentThread().getPriority());
		System.out.println("Thread 1 priority:"+t1.getPriority());
		System.out.println("Thread 2 priority:"+t2.getPriority());
}
}