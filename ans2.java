package assignment5;

class siblings
{
	String a;
	String b;
	
	void initialize(String a, String b)
	{
		this.a=a;
		this.b=b;
		//a=a; this kind of initialization prints null
		//b=b; this kind of initialization prints null
	}
	void printing()
	{
		System.out.println("Names of the siblings: "+a+" and "+b);
	}
}

public class ans2 {

	public static void main(String args[])
	{
		siblings chowdhry=new siblings();
		chowdhry.initialize("Aneeshaa", "Akshay");
		chowdhry.printing();
	}
}
