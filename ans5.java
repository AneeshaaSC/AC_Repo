package assignment8;

public class ans5 {

	public static void main(String args[])
	{
		String s="Women are women's ";
		String s1="worst enemy";
		StringBuilder s2= new StringBuilder(s1);
		StringBuffer s3=new StringBuffer(s1);
		
		//s1.append(s); doesn't even compile as string class objects are unmutable
		//s2.append(s);
		s2.insert(0, s);
		s3.append(s);
		System.out.println("s2:"+s2);
		System.out.println("s3:"+s3);
		s3.reverse();
		System.out.println("s3:"+s3);

		String s4=s2.toString();
		System.out.println("s4:"+s4+s4.getClass().getName());
		
		StringBuilder s5= new StringBuilder(30);
		StringBuffer s6=new StringBuffer(30);
		
		s5.append("Im unredeemable");
		s6.append("Im unredeemable");
		
		System.out.println("s5 length "+s5.length()+ " s5 capacity "+s5.capacity());
		System.out.println("s6 length "+s6.length()+ " s6 capacity "+s6.capacity());
		
		s5.append("I am unredeemable. This must end. Vicious circle. No Midas touch here. Hounded by ill luck.");
		System.out.println("s5 length "+s5.length()+ " s5 capacity "+s5.capacity());
		
		s5.ensureCapacity(50);
		System.out.println("s5 length "+s5.length()+ " s5 capacity "+s5.capacity());
		
		
	}
}
