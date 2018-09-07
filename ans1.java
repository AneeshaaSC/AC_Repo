class mydata
{
	String name;
	private String addr;
	//private int phoneno; //private member that is hidden from child class /objects
	void setvals(String a) 
	{
		addr=a;
	}
	String getvals()// indirectly accessing private variables
	{
		return addr;
	}
}

class mydata_work extends mydata
{
	String company,desig;
	
}


public class ans1 {
	public static void main(String args[]) 
	{
		mydata_work candidate1=new mydata_work();
		candidate1.name="Aneeshaa";
		candidate1.setvals("V'pura");
		String objaddr=candidate1.getvals();
		
		//candidate1.phoneno="9595";// this kind of access does not even allow successful compilation
		candidate1.company="GE";
		candidate1.desig="Data Engineer";
	
		System.out.println("Name: "+candidate1.name);
		System.out.println("Addr: "+objaddr);
		//System.out.println("phoneno: "+phoneno);
		System.out.println("company: "+candidate1.company);
		System.out.println("designation: "+candidate1.desig);
		
	}

}
